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
MAX_CONCURRENT_SANDBOXES = _k8s_limit if _k8s_limit > 0 else 150


sandbox_semaphore: Optional[asyncio.Semaphore] = None

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

    await asyncio.sleep(10)  # 等待预热窗口

    while True:
        try:
            loop = asyncio.get_running_loop()

            def _collect():
                nonlocal prev_net, prev_time

                # ---- 系统级 ----
                SYSTEM_METRICS["cpu_percent"].set(psutil.cpu_percent(interval=None))
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

            await loop.run_in_executor(None, _collect)
        except Exception as e:
            logger.error("System metrics collection error: %s", e)

        await asyncio.sleep(10)


@app.on_event("startup")
async def startup():
    global sandbox_semaphore
    sandbox_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SANDBOXES)
    logger.info(
        "Worker %d started, sandbox semaphore initialized with limit=%d",
        os.getpid(),
        MAX_CONCURRENT_SANDBOXES,
    )
    asyncio.create_task(_periodic_cleanup())
    asyncio.create_task(_collect_system_metrics())

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
    """kill 整个进程组（配合 start_new_session=True 使用）。
    确保子进程 fork 出的孙进程也被清理，防止孤儿进程堆积。
    os.killpg 是内核 syscall，立即返回，不阻塞 event loop。
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError, PermissionError):
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
        queue_timeout = req.compile_timeout + 10

        try:
            await asyncio.wait_for(sandbox_semaphore.acquire(), timeout=queue_timeout)
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
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin_data),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                _kill_proc_tree(proc)
                stdout, stderr = await proc.communicate()

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
                try:
                    await proc.wait()
                except Exception:
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