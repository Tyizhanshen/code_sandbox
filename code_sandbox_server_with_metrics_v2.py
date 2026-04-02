# ==============================
# 导入模块（按标准库、第三方库、本地模块分组，并排序）
# ==============================
import atexit
import json
import logging
import os
import pathlib
import resource
import shutil
import socket
import signal
import tempfile
import time
import uuid
import math
from typing import Dict, List, Optional
from functools import partial

import textwrap
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from starlette_exporter import PrometheusMiddleware
import asyncio
import psutil


_PROM_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
if not _PROM_DIR:
    _PROM_DIR = tempfile.mkdtemp(prefix="prom_multiproc_")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = _PROM_DIR
    logging.getLogger(__name__).warning(
        "PROMETHEUS_MULTIPROC_DIR was not set, using temp dir: %s", _PROM_DIR
    )


def _cleanup_stale_prom_files():
    """
    清除上一次运行残留的 .db 文件, 防止 Gauge livesum 在 gunicorn 重启后累加。
    使用 master PID marker 确保只在 gunicorn 首次启动时清理一次:
    - Worker 1 加载: 无 marker → 清理所有 .db → 写入 master PID
    - Worker 2~N 加载: marker 匹配 → 跳过
    - 下次重启: 新 master PID → 重新清理
    """
    if not os.path.isdir(_PROM_DIR):
        return

    marker = os.path.join(_PROM_DIR, ".master_pid")
    current_ppid = str(os.getppid())  # gunicorn master PID

    need_cleanup = True
    if os.path.exists(marker):
        try:
            with open(marker) as f:
                if f.read().strip() == current_ppid:
                    need_cleanup = False
        except Exception:
            pass

    if not need_cleanup:
        return

    for f in os.listdir(_PROM_DIR):
        if f.endswith(".db"):
            try:
                os.remove(os.path.join(_PROM_DIR, f))
            except OSError:
                pass

    try:
        with open(marker, "w") as f:
            f.write(current_ppid)
    except OSError:
        pass

    logging.getLogger(__name__).info(
        "Cleaned stale prometheus multiprocess files in %s (master_pid=%s)",
        _PROM_DIR, current_ppid,
    )


_cleanup_stale_prom_files()  # run at module load, before workers fork


def _cleanup_prom_dir():
    try:
        multiprocess.mark_process_dead(os.getpid())
    except Exception:
        pass


atexit.register(_cleanup_prom_dir)

# ==============================
# 工具：构建最小环境变量
# ==============================
def get_k8s_cpu_limit():
    try:
        quota = None
        period = None

        if os.path.isfile('/sys/fs/cgroup/cpu.max'):
            with open('/sys/fs/cgroup/cpu.max', 'r') as f:
                content = f.read().strip().split()
                if content[0] != 'max':
                    quota = int(content[0])
                    period = int(content[1])

        elif os.path.isfile('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'):
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
                quota = int(f.read().strip())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
                period = int(f.read().strip())

        # Quota / Period = cpu
        if quota is not None and period is not None and quota > 0:
            limit = math.ceil(quota / period)
            return int(limit)

    except Exception:
        raise

    return 0


_k8s_limit = get_k8s_cpu_limit()

# Semaphore: per-worker concurrency limit, configurable via env
_NUM_WORKERS = int(os.environ.get("SANDBOX_NUM_WORKERS", "12"))
_BASE_CPU = _k8s_limit if _k8s_limit > 0 else 90
_SEMAPHORE_LIMIT = int(os.environ.get(
    "SANDBOX_SEMAPHORE_LIMIT", str(max(_BASE_CPU * 6 // _NUM_WORKERS, 50))
))  # 90-core/12-worker → 50 per worker (global 600)


class MonitoredSemaphore:
    """
    带监控的信号量: 固定高上限, 提供 active 计数用于 Prometheus 指标。

    不做动态收缩 — 死循环/超时问题由 run_timeout + client disconnect detection 处理。
    降低并发只会让排队更严重, 不会缓解过载。
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._sem = asyncio.Semaphore(limit)
        self._active = 0

    async def acquire(self):
        await self._sem.acquire()
        self._active += 1

    def release(self):
        self._active -= 1
        self._sem.release()

    @property
    def active(self):
        return self._active


sandbox_semaphore: Optional[MonitoredSemaphore] = None

def build_sandbox_env() -> dict:
    keep_keys = [
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "PYTHONIOENCODING",
        "PYTHONPATH",
        "TMPDIR",
    ]
    env = {}
    for k in keep_keys:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v

    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
hostname = socket.gethostname()


DEFAULT_MAX_STDOUT = 4096
DEFAULT_MAX_STDERR = 4096
HARD_LIMIT_STDOUT = 65536
HARD_LIMIT_STDERR = 65536

BASE_WORK_DIR = "./tmp/ver1_firejail"
DEBUG_JSONL_DIR = "./tmp/ver1_firejail/debug_logs"

# ==============================
# JSONL 调试日志工具
# ==============================
ENABLE_DEBUG_JSONL = os.getenv("ENABLE_DEBUG_JSONL", "0").lower() in ("1", "true", "yes")
DEBUG_JSONL_MAX_SIZE = 50 * 1024 * 1024  # 50MB 单文件上限
DEBUG_JSONL_MAX_BACKUPS = 3              # 保留最近 3 个轮转文件


async def _write_debug_jsonl(entry: dict):
    if not ENABLE_DEBUG_JSONL:
        return
    try:
        os.makedirs(DEBUG_JSONL_DIR, exist_ok=True)
        filepath = os.path.join(DEBUG_JSONL_DIR, f"debug_{os.getpid()}.jsonl")
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _append_line_with_rotation, filepath, line)
    except Exception as e:
        logger.error(f"Failed to write debug JSONL: {e}")


def _append_line_with_rotation(filepath: str, line: str):
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > DEBUG_JSONL_MAX_SIZE:
            for i in range(DEBUG_JSONL_MAX_BACKUPS, 0, -1):
                src = f"{filepath}.{i}" if i > 1 else filepath
                dst = f"{filepath}.{i}" if i > 1 else f"{filepath}.1"
                if i == DEBUG_JSONL_MAX_BACKUPS:
                    old = f"{filepath}.{i}"
                    if os.path.exists(old):
                        os.remove(old)
                else:
                    if os.path.exists(src):
                        os.rename(src, f"{filepath}.{i + 1}")
            if os.path.exists(filepath):
                os.rename(filepath, f"{filepath}.1")
    except Exception:
        pass 

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line)

class RunReq(BaseModel):
    code: str
    language: str
    run_timeout: int = 200
    compile_timeout: int = 10
    stdin: Optional[str] = ""
    memory_limit_MB: int = 128

    files: Dict[str, str] = Field(default_factory=dict)
    fetch_files: List[str] = Field(default_factory=list)

    truncate_output: bool = True
    max_stdout: Optional[int] = None
    max_stderr: Optional[int] = None


# ==============================
# 监控指标定义（多进程模式）
# ==============================
CODE_METRICS = {
    "processing": Gauge(
        "code_processing_requests",
        "Current number of in-flight requests",
        ["path"],
        multiprocess_mode="livesum",
    ),
    "errors": Counter(
        "code_errors_total",
        "Total code execution errors",
        ["path", "error_type"],
    ),
    "timeouts": Counter(
        "code_timeouts_total",
        "Total timeout occurrences",
        ["path"],
    ),
    "successes": Counter(
        "code_successes_total",
        "Total successful executions",
        ["path"],
    ),
    "memory_usage": Gauge(
        "code_memory_usage_bytes",
        "Memory usage during execution (best-effort)",
        ["path"],
        multiprocess_mode="livesum",
    ),
    "duration": Histogram(
        "code_execution_duration_seconds",
        "Code execution duration in seconds",
        ["path"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60, 120, 300),
    ),
}

# ==============================
# 系统 & 进程级指标（替代 node_exporter / process_exporter）
# ==============================
SYSTEM_METRICS = {
    # ---- 系统级 ----
    "cpu_percent": Gauge(
        "sandbox_node_cpu_percent",
        "System CPU usage percentage",
        [],
        multiprocess_mode="livemax",
    ),
    "memory_percent": Gauge(
        "sandbox_node_memory_percent",
        "System memory usage percentage",
        [],
        multiprocess_mode="livemax",
    ),
    "disk_usage_percent": Gauge(
        "sandbox_node_disk_usage_percent",
        "Root filesystem disk usage percentage",
        [],
        multiprocess_mode="livemax",
    ),
    "net_recv_bps": Gauge(
        "sandbox_node_network_receive_bytes_per_sec",
        "Network receive rate (bytes/sec)",
        [],
        multiprocess_mode="livemax",
    ),
    "net_send_bps": Gauge(
        "sandbox_node_network_transmit_bytes_per_sec",
        "Network transmit rate (bytes/sec)",
        [],
        multiprocess_mode="livemax",
    ),
    # ---- 进程组级 ----
    "proc_cpu_percent": Gauge(
        "sandbox_process_cpu_percent",
        "Total CPU percent of all server processes",
        [],
        multiprocess_mode="livemax",
    ),
    "proc_memory_percent": Gauge(
        "sandbox_process_memory_percent",
        "Total memory percent of all server processes",
        [],
        multiprocess_mode="livemax",
    ),
    "proc_num_procs": Gauge(
        "sandbox_process_num_procs",
        "Number of server processes",
        [],
        multiprocess_mode="livemax",
    ),
    "proc_num_threads": Gauge(
        "sandbox_process_num_threads",
        "Total threads across all server processes",
        [],
        multiprocess_mode="livemax",
    ),
    "proc_open_fds": Gauge(
        "sandbox_process_open_fds",
        "Total open file descriptors across all server processes",
        [],
        multiprocess_mode="livemax",
    ),
    # ---- Change 4: Sandbox 并发 & 队列监控 ----
    "active_sandboxes": Gauge(
        "sandbox_active_sandboxes",
        "Number of currently running sandboxes on this worker",
        [],
        multiprocess_mode="livesum",
    ),
    "queue_waiting": Gauge(
        "sandbox_queue_waiting",
        "Number of requests waiting for semaphore",
        [],
        multiprocess_mode="livesum",
    ),
    "semaphore_limit": Gauge(
        "sandbox_semaphore_limit",
        "Configured semaphore limit for this worker (livesum = global total)",
        [],
        multiprocess_mode="livesum",
    ),
}

app = FastAPI()

custom_labels = {
    "service_type": "sandbox_service",
    "worker_node": hostname,
}

# starlette_exporter 自带的请求指标（QPS 等）
app.add_middleware(
    PrometheusMiddleware,
    labels=custom_labels,
    group_paths=True,
    skip_paths=["/metrics"],
)


async def _collect_system_metrics():
    """后台任务：每 10s 采集一次系统 & 进程组指标。"""
    # ---- warm-up: 首次调用 cpu_percent 返回 0，需要预热 ----
    psutil.cpu_percent(interval=None)
    prev_net = psutil.net_io_counters()
    prev_time = time.monotonic()

    # 预热进程级 cpu_percent
    try:
        cur_proc = psutil.Process()
        parent = cur_proc.parent()
        if parent and "gunicorn" in (parent.name() or ""):
            init_procs = [parent] + parent.children(recursive=True)
        else:
            init_procs = [cur_proc]
        for p in init_procs:
            try:
                p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass

    await asyncio.sleep(5)  # 等待预热窗口

    while True:
        try:
            loop = asyncio.get_running_loop()

            def _collect():
                nonlocal prev_net, prev_time

                # ---- 系统级 ----
                cpu_pct = psutil.cpu_percent(interval=None)
                SYSTEM_METRICS["cpu_percent"].set(cpu_pct)
                mem = psutil.virtual_memory()
                SYSTEM_METRICS["memory_percent"].set(mem.percent)

                try:
                    disk = psutil.disk_usage("/")
                    SYSTEM_METRICS["disk_usage_percent"].set(disk.percent)
                except OSError:
                    pass

                # 网络速率
                now = time.monotonic()
                cur_net = psutil.net_io_counters()
                dt = now - prev_time
                if dt > 0:
                    SYSTEM_METRICS["net_recv_bps"].set(
                        (cur_net.bytes_recv - prev_net.bytes_recv) / dt
                    )
                    SYSTEM_METRICS["net_send_bps"].set(
                        (cur_net.bytes_sent - prev_net.bytes_sent) / dt
                    )
                prev_net = cur_net
                prev_time = now

                # ---- 进程组级 ----
                try:
                    me = psutil.Process()
                    par = me.parent()
                    if par and "gunicorn" in (par.name() or ""):
                        all_procs = [par] + par.children(recursive=True)
                    else:
                        all_procs = [me]

                    total_cpu = 0.0
                    total_rss = 0
                    total_threads = 0
                    total_fds = 0
                    alive = 0

                    for p in all_procs:
                        try:
                            total_cpu += p.cpu_percent(interval=None)
                            total_rss += p.memory_info().rss
                            total_threads += p.num_threads()
                            total_fds += p.num_fds()
                            alive += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

                    mem_total = psutil.virtual_memory().total
                    SYSTEM_METRICS["proc_cpu_percent"].set(total_cpu)
                    SYSTEM_METRICS["proc_memory_percent"].set(
                        (total_rss / mem_total * 100) if mem_total > 0 else 0
                    )
                    SYSTEM_METRICS["proc_num_procs"].set(alive)
                    SYSTEM_METRICS["proc_num_threads"].set(total_threads)
                    SYSTEM_METRICS["proc_open_fds"].set(total_fds)
                except Exception as e:
                    logger.debug("Process metrics error: %s", e)

                return cpu_pct  # return for adaptive semaphore

            cpu_pct = await loop.run_in_executor(None, _collect)

            # Update sandbox metrics
            if sandbox_semaphore is not None:
                SYSTEM_METRICS["semaphore_limit"].set(sandbox_semaphore.limit)
                SYSTEM_METRICS["active_sandboxes"].set(sandbox_semaphore.active)

        except Exception as e:
            logger.error("System metrics collection error: %s", e)

        await asyncio.sleep(5)  # 5s 采样间隔


@app.on_event("startup")
async def startup():
    global sandbox_semaphore
    sandbox_semaphore = MonitoredSemaphore(limit=_SEMAPHORE_LIMIT)
    logger.info(
        "Worker %d started, semaphore limit=%d",
        os.getpid(),
        _SEMAPHORE_LIMIT,
    )
    asyncio.create_task(_periodic_cleanup())
    asyncio.create_task(_collect_system_metrics())
    asyncio.create_task(_reap_orphan_sandbox_procs())

_MAX_PROCESS_AGE = int(os.environ.get("MAX_PROCESS_AGE", 200))  # 5 min default

async def _reap_orphan_sandbox_procs():
    """Periodic background task that finds and kills leaked sandbox processes.

    Catches any child processes that escaped normal cleanup due to:
    - Race conditions in disconnect detection
    - Errors in _kill_proc_tree
    - User code spawning processes that escape the process group
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed, orphan reaper disabled")
        return

    while True:
        await asyncio.sleep(60)
        try:
            now = time.time()
            killed = 0
            for proc in psutil.process_iter(['pid', 'name', 'cwd', 'create_time', 'cmdline']):
                try:
                    info = proc.info
                    # Only target python processes
                    if info['name'] not in ('python', 'python3'):
                        continue
                    # Only target processes running in our sandbox workdir
                    cwd = info.get('cwd', '')
                    if not cwd or BASE_WORK_DIR not in cwd:
                        continue
                    # Only kill if running too long
                    age = now - info['create_time']
                    if age > _MAX_PROCESS_AGE:
                        cmdline = ' '.join(info.get('cmdline', []))[:100]
                        logger.warning(
                            "Killing orphan sandbox process pid=%s age=%ds cmd=%s",
                            info['pid'], int(age), cmdline
                        )
                        try:
                            proc.kill()
                            killed += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            if killed:
                logger.info("Reaped %d orphan sandbox processes", killed)
                CODE_METRICS["errors"].labels(
                    path="/system", error_type="orphan_reaped"
                ).inc(killed)
        except Exception as e:
            logger.error(f"Orphan reaper error: {e}")

async def _periodic_cleanup():
    while True:
        await asyncio.sleep(60)
        try:
            if os.path.exists(BASE_WORK_DIR):
                now = time.time()
                stale_count = 0
                for entry in os.scandir(BASE_WORK_DIR):
                    if entry.is_dir() and entry.name.startswith("fj_"):
                        try:
                            age = now - entry.stat().st_mtime
                            if age > 600:  # 10 分钟
                                shutil.rmtree(entry.path, ignore_errors=True)
                                stale_count += 1
                        except Exception:
                            pass
                if stale_count:
                    logger.info("Cleaned up %d stale temp directories", stale_count)
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")

@app.get("/metrics")
def metrics_endpoint():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    data = generate_latest(registry)
    return Response(content=data, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.middleware("http")
async def processing_gauge_middleware(request: Request, call_next):
    path = request.url.path
    if path in ("/model_code", "/custom_code", "/custom_code_A", "/custom_code_B", "/custom_code_C", "/custom_code_D", "/custom_code_E"):
        CODE_METRICS["processing"].labels(path=path).inc()
        try:
            return await call_next(request)
        finally:
            CODE_METRICS["processing"].labels(path=path).dec()
    return await call_next(request)


def set_memory_limit(mb_limit: int):
    if mb_limit <= 0:
        return
    soft = (mb_limit + 2048) * 1024 * 1024  # 给一点 buffer
    _, hard_as = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard_as))


def _kill_proc_tree(proc):
    """Kill the entire process tree rooted at proc.

    Strategy (ordered by reliability):
    1. psutil: recursively find ALL descendants (survives setsid/setpgrp)
    2. killpg: kill by process group (covers start_new_session=True)
    3. proc.kill(): last resort, kills only the direct child

    Each level is a fallback in case the previous one fails.
    """
    pid = proc.pid
    # Strategy 1: psutil (most reliable — finds children even if reparented)
    try:
        import psutil
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
            parent = None
        # Kill children first (bottom-up), then parent
        for child in reversed(children):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if parent:
            try:
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        # Wait for zombie reaping (up to 3s)
        gone, alive = psutil.wait_procs(children + ([parent] if parent else []), timeout=3)
        for p in alive:
            logger.warning("Process pid=%s still alive after SIGKILL+3s", p.pid)
        return
    except ImportError:
        pass  # psutil not installed, fall through

    # Strategy 2: killpg (process group)
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError, PermissionError):
        pass

    # Strategy 3: direct kill
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass

def decode_if_bytes(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x or ""


def truncate_output(text: str, max_length: int, default: int, hard_limit: int) -> str:
    limit = min(max_length or default, hard_limit)
    return text[:limit]


def _safe_join(base_dir: str, rel_path: str) -> str:
    """
    防目录穿越：确保 rel_path 落在 base_dir 下。
    - 禁止绝对路径
    - 禁止 .. 上跳
    """
    p = pathlib.PurePosixPath(rel_path)
    if p.is_absolute():
        raise ValueError(f"absolute path is not allowed: {rel_path}")
    if ".." in p.parts:
        raise ValueError(f"path traversal is not allowed: {rel_path}")

    base = pathlib.Path(base_dir).resolve()
    full = (pathlib.Path(base_dir) / rel_path).resolve()
    if not str(full).startswith(str(base) + os.sep):
        raise ValueError(f"path escapes base dir: {rel_path}")
    return str(full)


def prepare_environment(workdir: str, files: Dict[str, str]) -> None:
    """
    准备执行环境：
    - 支持多级目录
    - 防目录穿越
    """
    os.makedirs(workdir, exist_ok=True)
    for filename, content in (files or {}).items():
        fullpath = _safe_join(workdir, filename)
        parent = os.path.dirname(fullpath)
        os.makedirs(parent, exist_ok=True)
        with open(fullpath, "w", encoding="utf-8") as f:
            f.write(content)


def build_response(result: dict, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=result,
        status_code=status_code,
        headers={"X-Service-Node": hostname},
    )


def gen_user_code_in_try(user_code: str, error_marker: str) -> str:
    wrapped_user_code = textwrap.indent(user_code, " " * 4)
    return f"""#!/usr/bin/env python3
try:
{wrapped_user_code}
except Exception as e:
    raise Exception(f"{error_marker}" + repr(e) + f"{error_marker}")
"""

@app.post("/model_code")
async def model_code(request: Request, req: RunReq):
    return await _run_code(request, req)


@app.post("/custom_code")
async def custom_code(request: Request, req: RunReq):
    return await _run_code(request, req)

@app.post("/custom_code_A")
async def custom_code_A(request: Request, req: RunReq):
    return await _run_code(request, req)

@app.post("/custom_code_B")
async def custom_code_B(request: Request, req: RunReq):
    return await _run_code(request, req)

@app.post("/custom_code_C")
async def custom_code_C(request: Request, req: RunReq):
    return await _run_code(request, req)

@app.post("/custom_code_D")
async def custom_code_D(request: Request, req: RunReq):
    return await _run_code(request, req)

@app.post("/custom_code_E")
async def custom_code_E(request: Request, req: RunReq):
    return await _run_code(request, req)

async def _run_code(request: Request, req: RunReq):
    result = {"status": "", "run_status": "", "run_result": None}
    run_result = {}
    workdir = None
    start_time = None
    proc = None
    path = request.url.path
    status_code = 200

    try:
        # queue_timeout: 纯兜底安全网, 设为略大于 client_timeout (run_timeout + API_TIMEOUT=5)
        # 正常情况下, client 先超时断连 → disconnect detection 2s 内清理
        # 此值仅在 TCP 半开连接等极端情况下触发
        queue_timeout = (req.run_timeout or 30) + 5

        SYSTEM_METRICS["queue_waiting"].inc()
        try:
            # Race semaphore acquire against both timeout AND client disconnect.
            # Without disconnect detection here, a retrying client's abandoned
            # request would still wait for + occupy a semaphore slot, wasting
            # server capacity and causing queue buildup.
            acquired = False
            sem_task = asyncio.ensure_future(sandbox_semaphore.acquire())
            deadline = asyncio.get_event_loop().time() + queue_timeout
            try:
                while not sem_task.done():
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        sem_task.cancel()
                        try:
                            await sem_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        raise asyncio.TimeoutError()

                    # Wait for semaphore for up to 2s, then check disconnect
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(sem_task), timeout=min(2.0, remaining)
                        )
                    except asyncio.TimeoutError:
                        pass  # Check disconnect below

                    if sem_task.done():
                        break

                    # Check if client disconnected while we were waiting
                    try:
                        if await request.is_disconnected():
                            # Race: semaphore might have been acquired between
                            # the shield timeout and this check. Must release!
                            if sem_task.done() and not sem_task.cancelled():
                                try:
                                    sandbox_semaphore.release()
                                except Exception:
                                    pass
                            else:
                                sem_task.cancel()
                                try:
                                    await sem_task
                                except (asyncio.CancelledError, Exception):
                                    pass
                            logger.debug(
                                "Client disconnected during queue wait, discarding request"
                            )
                            CODE_METRICS["errors"].labels(
                                path=path, error_type="client_disconnected"
                            ).inc()
                            run_result.update({
                                "status": "ClientDisconnected",
                                "stderr": "Client disconnected while waiting in queue",
                                "execution_time": 0,
                            })
                            result.update({"status": "Aborted", "run_result": run_result})
                            return build_response(result, 499)
                    except Exception:
                        pass  # Can't check disconnect, keep waiting

                acquired = True
            except:
                if sem_task.done() and not sem_task.cancelled():
                    # We acquired but hit an error; release to avoid leak
                    try:
                        sandbox_semaphore.release()
                    except Exception:
                        pass
                raise

        except asyncio.TimeoutError:
            logger.warning(
                "Semaphore queue timeout after %ds, discarding request (client likely already disconnected)",
                queue_timeout,
            )
            CODE_METRICS["errors"].labels(path=path, error_type="queue_timeout").inc()
            run_result.update({
                "status": "QueueTimeout",
                "stderr": f"Server queue timeout after {queue_timeout}s: too many concurrent requests",
                "execution_time": 0,
            })
            result.update({"status": "SandboxError", "run_result": run_result})
            return build_response(result, 504)
        finally:
            SYSTEM_METRICS["queue_waiting"].dec()

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, partial(os.makedirs, BASE_WORK_DIR, exist_ok=True))
            workdir = await loop.run_in_executor(
                None, partial(tempfile.mkdtemp, prefix="fj_", dir=BASE_WORK_DIR)
            )

            # 2) files
            await loop.run_in_executor(None, prepare_environment, workdir, req.files)

            # 3) main.py
            script_path = os.path.join(workdir, "main.py")
            error_marker = str(uuid.uuid4())
            code_content = gen_user_code_in_try(req.code, error_marker)

            def _write_script():
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(code_content)

            await loop.run_in_executor(None, _write_script)

            # 4) cmd
            cmd = [req.language, "main.py"]
            sandbox_env = build_sandbox_env()

            # 5) exec asyncio.create_subprocess_exec
            start_time = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workdir,
                env=sandbox_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if req.stdin else None,
                preexec_fn=partial(set_memory_limit, req.memory_limit_MB),
                start_new_session=True,
            )

            try:
                stdin_data = (req.stdin or "").encode() if req.stdin else None
                timeout = (req.run_timeout or 0) + 1

                # Change 3: background task to detect client disconnect
                # Uses a lightweight background coroutine instead of inline polling
                # to avoid event loop contention under high concurrency.
                _disconnect_detected = False

                async def _check_disconnect():
                    nonlocal _disconnect_detected
                    while True:
                        await asyncio.sleep(5)  # check every 5s, lightweight
                        try:
                            if await request.is_disconnected():
                                _disconnect_detected = True
                                _kill_proc_tree(proc)
                                return
                        except Exception:
                            return

                disconnect_task = asyncio.create_task(_check_disconnect())

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=stdin_data),
                        timeout=timeout,
                    )
                finally:
                    disconnect_task.cancel()
                    try:
                        await disconnect_task
                    except (asyncio.CancelledError, Exception):
                        pass

                # If process was killed due to client disconnect
                if _disconnect_detected:
                    CODE_METRICS["errors"].labels(
                        path=path, error_type="client_disconnected"
                    ).inc()
                    duration = time.monotonic() - start_time
                    CODE_METRICS["duration"].labels(path=path).observe(duration)
                    run_result.update({
                        "status": "ClientDisconnected",
                        "stderr": "Client disconnected during execution",
                        "execution_time": duration,
                    })
                    result.update({"status": "Aborted", "run_result": run_result})
                    return build_response(result, 499)


            except asyncio.TimeoutError:
                # Change 5: kill then wait (don't read pipe)
                _kill_proc_tree(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Process pid=%s did not exit after SIGKILL+5s",
                        proc.pid,
                    )
                stdout, stderr = b"", b"TimeLimitExceeded"

                duration = time.monotonic() - start_time
                CODE_METRICS["timeouts"].labels(path=path).inc()
                CODE_METRICS["errors"].labels(path=path, error_type="code_run_timeout").inc()
                CODE_METRICS["duration"].labels(path=path).observe(duration)

                run_result.update(
                    {
                        "status": "TimeLimitExceeded",
                        "stderr": "TimeLimitExceeded",
                        "execution_time": duration,
                    }
                )
                result.update({"status": "Failed", "run_result": run_result})
                return build_response(result, 200)
            except (RuntimeError, BrokenPipeError, ConnectionResetError) as e:
                logger.warning(f"Subprocess stdin pipe error: {e}")
                _kill_proc_tree(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (asyncio.TimeoutError, Exception):
                    pass
                stdout = b""
                stderr = str(e).encode()

            # 6) decode & truncate
            stdout = decode_if_bytes(stdout)
            stderr = decode_if_bytes(stderr)

            if req.truncate_output:
                stdout = truncate_output(stdout, req.max_stdout, DEFAULT_MAX_STDOUT, HARD_LIMIT_STDOUT)
                stderr = truncate_output(stderr, req.max_stderr, DEFAULT_MAX_STDERR, HARD_LIMIT_STDERR)

            duration = time.monotonic() - start_time
            CODE_METRICS["duration"].labels(path=path).observe(duration)

            run_result.update(
                {
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": proc.returncode if proc.returncode is not None else -1,
                    "return_code": proc.returncode,
                    "execution_time": duration,
                }
            )

            if proc.returncode == 0:
                result["status"] = "Success"
                run_result["status"] = "Finished"
                CODE_METRICS["successes"].labels(path=path).inc()
            else:
                result["status"] = "Failed"
                combined = stdout + stderr

                if error_marker in combined:
                    run_result["stdout"] = stdout.replace(error_marker, "")
                    if "MemoryError" in stderr:
                        run_result["status"] = "MemoryLimitExceeded"
                        CODE_METRICS["errors"].labels(path=path, error_type="code_mem_err").inc()
                    else:
                        run_result["status"] = "Finished"
                        CODE_METRICS["errors"].labels(path=path, error_type="code_run_err").inc()
                else:
                    run_result["status"] = "Finished"
                    CODE_METRICS["errors"].labels(path=path, error_type="code_run_err").inc()

            result["run_result"] = run_result
            return build_response(result, 200)

        finally:
            sandbox_semaphore.release()

    except Exception as e:
        import traceback as tb
        logger.error(f"Execution error: {str(e)}\n{tb.format_exc()}")

        duration = time.monotonic() - start_time if start_time else 0.0
        CODE_METRICS["errors"].labels(path=path, error_type="web_server_exception").inc()
        if start_time:
            CODE_METRICS["duration"].labels(path=path).observe(duration)

        run_result.update(
            {
                "status": "InternalError",
                "stderr": f"An unexpected internal error occurred: {e}",
                "execution_time": duration,
            }
        )
        result.update({"status": "SandboxError", "run_result": run_result})
        return build_response(result, 500)

    finally:
        # log
        try:
            if proc is not None and proc.returncode is None:
                _kill_proc_tree(proc)
                try:
                    await proc.wait()
                except Exception:
                    pass
                logger.warning("Cleaned up leaked subprocess pid=%s", getattr(proc, 'pid', '?'))
        except Exception as e:
            logger.error(f"Subprocess cleanup failed: {e}")
        try:
            log_entry = {
                "timestamp": time.time(),
                "worker_pid": os.getpid(),
                "request": {
                    "path": path,
                    "language": getattr(req, "language", None),
                    "run_timeout": getattr(req, "run_timeout", None),
                    "memory_limit_MB": getattr(req, "memory_limit_MB", None),
                    "stdin": (getattr(req, "stdin", "") or "")[:500],
                    "code": (getattr(req, "code", "") or "")[:1000],
                },
                "response": {
                    "http_status": status_code,
                    "status": result.get("status"),
                    "run_status": run_result.get("status"),
                    "return_code": run_result.get("return_code"),
                    "execution_time": run_result.get("execution_time"),
                    "stdout_preview": (run_result.get("stdout") or "")[:500],
                    "stderr_preview": (run_result.get("stderr") or "")[:500],
                },
            }
            logger.info("Execution log: %s", json.dumps(log_entry, ensure_ascii=False))
            #jsonl debug log
            await _write_debug_jsonl(log_entry)
        except Exception as e:
            logger.error(f"Logging failed: {str(e)}")

        # cleanup
        if workdir and os.path.exists(workdir):
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, shutil.rmtree, workdir, True)
            except Exception as e:
                logger.error(f"Cleanup failed: {str(e)}")



@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}