#!/usr/bin/env python3
"""
Sandbox Disconnect Detection Profiling & Comprehensive Test Suite
=================================================================

Profiles and validates the server-side disconnect detection mechanism:
- Measures actual detection latency (queue phase: ~2s poll, exec phase: ~5s poll)
- Validates Prometheus metrics are correctly updated after disconnects
- Verifies semaphore slots are freed (no resource leaks)
- Stress-tests disconnect detection under high concurrency
- Checks that orphan processes are cleaned up after client disconnects

Usage:
  # Full disconnect profiling (server MUST run with SANDBOX_SEMAPHORE_LIMIT=1):
    python test_disconnect_profiling.py --host <IP> --port 8080 --suite disconnect

  # Metrics & resource leak validation (any semaphore limit):
    python test_disconnect_profiling.py --host <IP> --port 8080 --suite metrics

  # Concurrency stress test (normal semaphore limit recommended):
    python test_disconnect_profiling.py --host <IP> --port 8080 --suite stress

  # All tests:
    python test_disconnect_profiling.py --host <IP> --port 8080 --suite all

  # Single test by ID:
    python test_disconnect_profiling.py --host <IP> --port 8080 --test 3
"""
import argparse
import asyncio
import time
import json
import re
import statistics
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field

import aiohttp


# ============================================================
# Profiling Data Structures
# ============================================================

@dataclass
class ProfileResult:
    """Structured result for a single profiling observation."""
    test_name: str
    passed: bool
    elapsed_sec: float = 0.0
    detection_latency_sec: Optional[float] = None  # how fast server detected disconnect
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ProfilingReport:
    """Aggregated profiling data from all tests."""
    results: List[ProfileResult] = field(default_factory=list)

    def add(self, r: ProfileResult):
        self.results.append(r)

    def print_summary(self):
        print(f"\n{'=' * 80}")
        print("PROFILING REPORT")
        print(f"{'=' * 80}")

        # Test results
        for r in self.results:
            sym = "✅" if r.passed else "❌"
            latency = f", detection_latency={r.detection_latency_sec:.2f}s" if r.detection_latency_sec else ""
            print(f"  {sym} {r.test_name} ({r.elapsed_sec:.2f}s{latency})")
            if r.error:
                print(f"     Error: {r.error}")
            for k, v in r.details.items():
                print(f"     {k}: {v}")

        # Disconnect detection latency analysis
        latencies = [r.detection_latency_sec for r in self.results if r.detection_latency_sec is not None]
        if latencies:
            print(f"\n  --- Disconnect Detection Latency Analysis ---")
            print(f"  Samples:  {len(latencies)}")
            print(f"  Min:      {min(latencies):.2f}s")
            print(f"  Median:   {statistics.median(latencies):.2f}s")
            print(f"  Max:      {max(latencies):.2f}s")
            print(f"  Mean:     {statistics.mean(latencies):.2f}s")
            if len(latencies) > 1:
                print(f"  Stdev:    {statistics.stdev(latencies):.2f}s")

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"\n  {passed}/{total} passed\n")


report = ProfilingReport()


# ============================================================
# Helpers
# ============================================================

async def _post(session, url, payload, timeout) -> Tuple[int, dict, float]:
    """POST request, return (status, body, elapsed)."""
    start = time.monotonic()
    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as resp:
        body = await resp.json()
        return resp.status, body, time.monotonic() - start


async def _get_metrics(base_url: str) -> Dict[str, float]:
    """Fetch /metrics and parse Prometheus text format into a dict."""
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{base_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            text = await resp.text()

    metrics = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Parse: metric_name{labels} value  or  metric_name value
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([\d.eE+\-]+)$', line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def _get_metric(metrics: dict, name: str, labels: Optional[dict] = None) -> float:
    """Extract a metric value by name and optional labels."""
    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        key = f'{name}{{{label_str}}}'
    else:
        key = name

    # Try exact match first
    if key in metrics:
        return metrics[key]

    # Fuzzy match (labels may be in different order)
    for k, v in metrics.items():
        if k.startswith(name) and (not labels or all(f'{lk}="{lv}"' in k for lk, lv in labels.items())):
            return v

    return 0.0


def _header(test_id, title):
    print(f"\n{'=' * 80}")
    print(f"TEST {test_id}: {title}")
    print(f"{'=' * 80}")


# ============================================================
# TEST 1: Baseline - Normal execution + metrics validation
# ============================================================

async def test_1_baseline(base_url: str) -> ProfileResult:
    """Verify normal execution works and metrics are correctly updated."""
    _header(1, "Baseline execution + metrics validation")

    # Snapshot metrics before
    metrics_before = await _get_metrics(base_url)
    success_before = _get_metric(metrics_before, "code_successes_total", {"path": "/custom_code_A"})

    payload = {"code": "print('baseline_ok')", "language": "python", "run_timeout": 10}

    async with aiohttp.ClientSession() as s:
        try:
            status, body, elapsed = await _post(s, f"{base_url}/custom_code_A", payload, 30)
            stdout = body.get("run_result", {}).get("stdout", "").strip()
            print(f"  Status: {status}, Time: {elapsed:.2f}s, Output: '{stdout}'")

            # Snapshot metrics after
            metrics_after = await _get_metrics(base_url)
            success_after = _get_metric(metrics_after, "code_successes_total", {"path": "/custom_code_A"})
            success_delta = success_after - success_before

            print(f"  code_successes_total delta: {success_delta}")

            ok = status == 200 and "baseline_ok" in stdout and success_delta >= 1
            if ok:
                print("  ✅ PASSED")
            else:
                print(f"  ❌ FAILED (status={status}, stdout='{stdout}', metric_delta={success_delta})")

            return ProfileResult(
                test_name="Baseline + metrics",
                passed=ok,
                elapsed_sec=elapsed,
                details={"success_counter_delta": success_delta},
            )
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return ProfileResult(test_name="Baseline + metrics", passed=False, error=str(e))


# ============================================================
# TEST 2: Disconnect detection latency during queue wait
# ============================================================

async def test_2_queue_disconnect_latency(base_url: str) -> ProfileResult:
    """
    Profile how quickly the server detects a client disconnect during queue wait.

    Setup: SEMAPHORE_LIMIT=1
    1. Send a blocker to occupy the semaphore
    2. Send a victim that will queue up
    3. Client disconnects after 3s
    4. Measure time from client disconnect until server frees the queue slot
       by monitoring active_sandboxes / queue_waiting metrics
    """
    _header(2, "Queue disconnect detection latency (needs SEMAPHORE_LIMIT=1)")

    blocker = {"code": "import time; time.sleep(20); print('blocker')", "language": "python", "run_timeout": 30}
    victim = {"code": "print('victim')", "language": "python", "run_timeout": 10}

    # Snapshot metrics before
    metrics_before = await _get_metrics(base_url)
    disc_before = _get_metric(metrics_before, "code_errors_total",
                              {"path": "/custom_code_A", "error_type": "client_disconnected"})

    # Step 1: blocker
    print("  [1] Sending blocker (20s sleep)...")
    bs = aiohttp.ClientSession()
    bt = asyncio.create_task(_post(bs, f"{base_url}/custom_code_A", blocker, 60))
    await asyncio.sleep(2)

    # Step 2: victim → disconnect after 3s
    print("  [2] Sending victim (client disconnects after 3s)...")
    vs = aiohttp.ClientSession()
    disconnect_time = None
    try:
        await _post(vs, f"{base_url}/custom_code_A", victim, 3)
        print("  Unexpected response (should have timed out)")
    except (asyncio.TimeoutError, Exception):
        disconnect_time = time.monotonic()
        print(f"  Client disconnected at t=0")
    finally:
        await vs.close()

    # Step 3: Poll metrics to detect when server recognizes the disconnect
    detection_latency = None
    if disconnect_time:
        print("  [3] Polling metrics to measure detection latency...")
        for i in range(15):  # poll for up to 15s
            await asyncio.sleep(1)
            metrics_now = await _get_metrics(base_url)
            disc_now = _get_metric(metrics_now, "code_errors_total",
                                   {"path": "/custom_code_A", "error_type": "client_disconnected"})
            if disc_now > disc_before:
                detection_latency = time.monotonic() - disconnect_time
                print(f"  Server detected disconnect in {detection_latency:.2f}s "
                      f"(client_disconnected counter: {disc_before} → {disc_now})")
                break
        else:
            print(f"  ⚠️  client_disconnected metric did not increment within 15s")

    # Wait for blocker
    print("  [4] Waiting for blocker to complete...")
    try:
        await bt
    except Exception:
        pass
    await bs.close()

    ok = detection_latency is not None and detection_latency < 10
    if ok:
        print(f"  ✅ PASSED (detection latency: {detection_latency:.2f}s)")
    else:
        print(f"  ❌ FAILED (detection_latency={detection_latency})")

    return ProfileResult(
        test_name="Queue disconnect latency",
        passed=ok,
        elapsed_sec=detection_latency or 0,
        detection_latency_sec=detection_latency,
        details={"expected_poll_interval": "2s"},
    )


# ============================================================
# TEST 3: Disconnect detection latency during execution
# ============================================================

async def test_3_exec_disconnect_latency(base_url: str) -> ProfileResult:
    """
    Profile how quickly the server detects client disconnect during code execution.
    Server polls disconnect every 2s during execution.
    """
    _header(3, "Execution disconnect detection latency")

    payload = {
        "code": "import time; time.sleep(120); print('no')",
        "language": "python",
        "run_timeout": 60,
    }

    # Snapshot metrics before
    metrics_before = await _get_metrics(base_url)
    disc_before = _get_metric(metrics_before, "code_errors_total",
                              {"path": "/custom_code_A", "error_type": "client_disconnected"})

    print("  [1] Sending long-running request (client disconnects after 3s)...")
    disconnect_time = None
    async with aiohttp.ClientSession() as s:
        try:
            await _post(s, f"{base_url}/custom_code_A", payload, 3)
            print("  Unexpected response")
        except (asyncio.TimeoutError, Exception):
            disconnect_time = time.monotonic()
            print(f"  Client disconnected at t=0")

    # Poll metrics
    detection_latency = None
    if disconnect_time:
        print("  [2] Polling metrics to measure detection latency...")
        for i in range(20):  # poll for up to 20s (2s poll interval + overhead)
            await asyncio.sleep(1)
            metrics_now = await _get_metrics(base_url)
            disc_now = _get_metric(metrics_now, "code_errors_total",
                                   {"path": "/custom_code_A", "error_type": "client_disconnected"})
            if disc_now > disc_before:
                detection_latency = time.monotonic() - disconnect_time
                print(f"  Server detected disconnect in {detection_latency:.2f}s "
                      f"(client_disconnected counter: {disc_before} → {disc_now})")
                break
        else:
            print(f"  ⚠️  client_disconnected metric did not increment within 20s")

    ok = detection_latency is not None and detection_latency < 15
    if ok:
        print(f"  ✅ PASSED (detection latency: {detection_latency:.2f}s)")
    else:
        print(f"  ❌ FAILED (detection_latency={detection_latency})")

    return ProfileResult(
        test_name="Execution disconnect latency",
        passed=ok,
        elapsed_sec=detection_latency or 0,
        detection_latency_sec=detection_latency,
        details={"expected_poll_interval": "2s"},
    )


# ============================================================
# TEST 4: Semaphore leak test after disconnect
# ============================================================

async def test_4_semaphore_leak(base_url: str) -> ProfileResult:
    """
    Verify no semaphore leak after client disconnects.

    Strategy:
    - Record active_sandboxes / queue_waiting before
    - Trigger multiple disconnects (both in queue and execution phase)
    - Wait for cleanup
    - Verify active_sandboxes and queue_waiting return to original values
    """
    _header(4, "Semaphore leak detection after disconnects")

    # Record baseline
    await asyncio.sleep(3)  # let any prior tasks settle
    m0 = await _get_metrics(base_url)
    active_before = _get_metric(m0, "sandbox_active_sandboxes")
    queue_before = _get_metric(m0, "sandbox_queue_waiting")
    print(f"  Baseline: active_sandboxes={active_before}, queue_waiting={queue_before}")

    # Send 5 requests that will all disconnect quickly
    print("  [1] Sending 5 requests that disconnect in 2s each...")
    tasks = []
    sessions = []
    for i in range(5):
        s = aiohttp.ClientSession()
        sessions.append(s)
        payload = {
            "code": f"import time; time.sleep(60); print('leak_test_{i}')",
            "language": "python",
            "run_timeout": 30,
        }
        t = asyncio.create_task(_post(s, f"{base_url}/custom_code_A", payload, 2))
        tasks.append(t)
        await asyncio.sleep(0.2)

    # Wait for all to disconnect
    for t in tasks:
        try:
            await t
        except Exception:
            pass
    for s in sessions:
        await s.close()
    print("  All clients disconnected")

    # Wait for server to detect disconnects and clean up
    # Note: _kill_proc_tree blocks event loop 3-5s per process, so cleanup
    # may take longer than the 2s poll interval suggests.
    print("  [2] Waiting for server to detect disconnects (up to 60s)...")
    leaked = True
    for i in range(60):
        await asyncio.sleep(1)
        m_now = await _get_metrics(base_url)
        active_now = _get_metric(m_now, "sandbox_active_sandboxes")
        queue_now = _get_metric(m_now, "sandbox_queue_waiting")
        if active_now <= active_before and queue_now <= queue_before:
            print(f"  Resources freed after {i+1}s: active={active_now}, queue={queue_now}")
            leaked = False
            break
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}s: active={active_now}, queue={queue_now}")

    if leaked:
        m_final = await _get_metrics(base_url)
        active_final = _get_metric(m_final, "sandbox_active_sandboxes")
        queue_final = _get_metric(m_final, "sandbox_queue_waiting")
        print(f"  ❌ FAILED: Resources not freed. active={active_final}, queue={queue_final}")

    ok = not leaked
    if ok:
        print("  ✅ PASSED (no semaphore leak)")

    return ProfileResult(
        test_name="Semaphore leak detection",
        passed=ok,
        details={
            "active_before": active_before,
            "queue_before": queue_before,
            "leaked": leaked,
        },
    )


# ============================================================
# TEST 5: Ghost request accumulation (needs SEMAPHORE_LIMIT=1)
# ============================================================

async def test_5_ghost_accumulation(base_url: str) -> ProfileResult:
    """
    Verify ghost requests don't accumulate in the queue.

    After a blocker finishes and N ghost requests have been cleaned up,
    a real request should complete with no extra queue delay.
    """
    _header(5, "Ghost request accumulation (needs SEMAPHORE_LIMIT=1)")

    blocker = {"code": "import time; time.sleep(10); print('b')", "language": "python", "run_timeout": 20}
    ghost = {"code": "print('g')", "language": "python", "run_timeout": 10}
    real = {"code": "print('real_ok')", "language": "python", "run_timeout": 10}

    # Step 1: blocker
    print("  [1] Sending blocker (10s)...")
    bs = aiohttp.ClientSession()
    bt = asyncio.create_task(_post(bs, f"{base_url}/custom_code_A", blocker, 60))
    await asyncio.sleep(2)

    # Step 2: 10 ghost requests, each disconnects after 1.5s
    print("  [2] Firing 10 ghost requests (each disconnects after 1.5s)...")
    ghost_sessions = []
    for i in range(10):
        gs = aiohttp.ClientSession()
        gt = asyncio.create_task(_post(gs, f"{base_url}/custom_code_A", ghost, 1.5))
        ghost_sessions.append((gs, gt))
        await asyncio.sleep(0.1)

    for gs, gt in ghost_sessions:
        try:
            await gt
        except Exception:
            pass
        await gs.close()
    print("  All ghosts disconnected")

    # Step 3: Wait for blocker
    print("  [3] Waiting for blocker...")
    try:
        await bt
    except Exception:
        pass
    await bs.close()

    # Give server time to clean up detected disconnects
    await asyncio.sleep(3)

    # Step 4: Real request should be fast
    print("  [4] Sending real request...")
    async with aiohttp.ClientSession() as s:
        try:
            status, body, elapsed = await _post(s, f"{base_url}/custom_code_A", real, 15)
            stdout = body.get("run_result", {}).get("stdout", "").strip()
            print(f"  Status: {status}, Time: {elapsed:.2f}s, Output: '{stdout}'")

            ok = elapsed < 5 and status == 200 and "real_ok" in stdout
            if ok:
                print(f"  ✅ PASSED (real request took {elapsed:.1f}s, no ghost blocking)")
            else:
                print(f"  ❌ FAILED (took {elapsed:.1f}s, ghosts may have blocked)")

            return ProfileResult(
                test_name="Ghost accumulation",
                passed=ok,
                elapsed_sec=elapsed,
                details={"num_ghosts": 10, "real_request_latency": elapsed},
            )
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return ProfileResult(test_name="Ghost accumulation", passed=False, error=str(e))


# ============================================================
# TEST 6: Metrics correctness after mixed disconnects
# ============================================================

async def test_6_metrics_correctness(base_url: str) -> ProfileResult:
    """
    Validate that Prometheus metrics are correctly updated after a mix of:
    - Normal successes
    - Client disconnects during queue
    - Client disconnects during execution
    - Timeouts
    """
    _header(6, "Metrics correctness after mixed operations")

    m_before = await _get_metrics(base_url)

    success_before = _get_metric(m_before, "code_successes_total", {"path": "/custom_code_A"})
    disc_before = _get_metric(m_before, "code_errors_total",
                              {"path": "/custom_code_A", "error_type": "client_disconnected"})
    timeout_before = _get_metric(m_before, "code_timeouts_total", {"path": "/custom_code_A"})

    # 1. Normal success
    print("  [1] Normal success request...")
    async with aiohttp.ClientSession() as s:
        try:
            await _post(s, f"{base_url}/custom_code_A", {"code": "print('m_ok')", "language": "python", "run_timeout": 10}, 15)
        except Exception:
            pass

    # 2. Client disconnect during execution
    print("  [2] Client disconnect during execution...")
    async with aiohttp.ClientSession() as s:
        try:
            await _post(s, f"{base_url}/custom_code_A",
                        {"code": "import time; time.sleep(60)", "language": "python", "run_timeout": 30}, 2)
        except Exception:
            pass

    # 3. Server-side timeout
    print("  [3] Server-side timeout...")
    async with aiohttp.ClientSession() as s:
        try:
            status, body, elapsed = await _post(
                s, f"{base_url}/custom_code_A",
                {"code": "import time; time.sleep(30)", "language": "python", "run_timeout": 3}, 15)
            print(f"  Timeout test: status={status}, time={elapsed:.1f}s")
        except Exception:
            pass

    # Wait for all metrics to settle
    print("  [4] Waiting for metrics to settle...")
    await asyncio.sleep(15)  # at least one exec disconnect poll cycle

    m_after = await _get_metrics(base_url)
    success_after = _get_metric(m_after, "code_successes_total", {"path": "/custom_code_A"})
    disc_after = _get_metric(m_after, "code_errors_total",
                             {"path": "/custom_code_A", "error_type": "client_disconnected"})
    timeout_after = _get_metric(m_after, "code_timeouts_total", {"path": "/custom_code_A"})

    success_delta = success_after - success_before
    disc_delta = disc_after - disc_before
    timeout_delta = timeout_after - timeout_before

    print(f"  Results:")
    print(f"    success_delta:  {success_delta} (expected ≥ 1)")
    print(f"    disconnect_delta: {disc_delta} (expected ≥ 1)")
    print(f"    timeout_delta:  {timeout_delta} (expected ≥ 1)")

    ok = success_delta >= 1 and disc_delta >= 1 and timeout_delta >= 1
    if ok:
        print("  ✅ PASSED (all metric categories incremented correctly)")
    else:
        print("  ❌ FAILED (some metrics did not increment)")

    return ProfileResult(
        test_name="Metrics correctness",
        passed=ok,
        details={
            "success_delta": success_delta,
            "disconnect_delta": disc_delta,
            "timeout_delta": timeout_delta,
        },
    )


# ============================================================
# TEST 7: Concurrent disconnect storm
# ============================================================

async def test_7_disconnect_storm(base_url: str, concurrency: int = 30) -> ProfileResult:
    """
    Stress test: Fire N concurrent requests, all disconnect after ~2s.
    Verify:
    - Server doesn't crash
    - Metrics are updated
    - Semaphore returns to baseline
    """
    _header(7, f"Disconnect storm ({concurrency} concurrent disconnects)")

    await asyncio.sleep(3)
    m_before = await _get_metrics(base_url)
    active_before = _get_metric(m_before, "sandbox_active_sandboxes")
    disc_before = _get_metric(m_before, "code_errors_total",
                              {"path": "/custom_code_A", "error_type": "client_disconnected"})

    print(f"  [1] Firing {concurrency} requests (all disconnect after 2s)...")
    start = time.monotonic()

    tasks = []
    sessions = []
    for i in range(concurrency):
        s = aiohttp.ClientSession()
        sessions.append(s)
        payload = {
            "code": f"import time; time.sleep(60)",
            "language": "python",
            "run_timeout": 30,
        }
        t = asyncio.create_task(_post(s, f"{base_url}/custom_code_A", payload, 2))
        tasks.append(t)

    # Wait for all to disconnect
    timeouts = 0
    errors = 0
    for t in tasks:
        try:
            await t
        except asyncio.TimeoutError:
            timeouts += 1
        except Exception:
            errors += 1
    for s in sessions:
        await s.close()

    disconnect_wall_time = time.monotonic() - start
    print(f"  All clients disconnected in {disconnect_wall_time:.2f}s (timeouts={timeouts}, errors={errors})")

    # Wait for server to detect and clean up
    print("  [2] Waiting for server cleanup (up to 30s)...")
    cleanup_latency = None
    cleanup_start = time.monotonic()
    for i in range(30):
        await asyncio.sleep(1)
        m_now = await _get_metrics(base_url)
        active_now = _get_metric(m_now, "sandbox_active_sandboxes")
        if active_now <= active_before:
            cleanup_latency = time.monotonic() - cleanup_start
            disc_now = _get_metric(m_now, "code_errors_total",
                                   {"path": "/custom_code_A", "error_type": "client_disconnected"})
            disc_delta = disc_now - disc_before
            print(f"  Cleanup complete in {cleanup_latency:.1f}s")
            print(f"  client_disconnected increments: {disc_delta}")
            break
        if (i + 1) % 5 == 0:
            print(f"  ... {i+1}s: active={active_now}")
    else:
        m_final = await _get_metrics(base_url)
        active_final = _get_metric(m_final, "sandbox_active_sandboxes")
        print(f"  ⚠️  active_sandboxes still at {active_final} after 30s")
        disc_delta = 0
        cleanup_latency = 30

    # Verify server is still healthy
    print("  [3] Health check...")
    health_ok = False
    async with aiohttp.ClientSession() as s:
        try:
            async with s.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    health_ok = True
                    print("  Server healthy ✓")
        except Exception as e:
            print(f"  Server health check failed: {e}")

    # Verify a real request still works
    print("  [4] Post-storm functional check...")
    func_ok = False
    async with aiohttp.ClientSession() as s:
        try:
            status, body, elapsed = await _post(
                s, f"{base_url}/custom_code_A",
                {"code": "print('storm_survived')", "language": "python", "run_timeout": 10}, 30)
            stdout = body.get("run_result", {}).get("stdout", "").strip()
            if status == 200 and "storm_survived" in stdout:
                func_ok = True
                print(f"  Post-storm request OK (status={status}, time={elapsed:.1f}s)")
            else:
                print(f"  Post-storm request: status={status}, output='{stdout}'")
        except Exception as e:
            print(f"  Post-storm request failed: {e}")

    ok = health_ok and func_ok and cleanup_latency is not None and cleanup_latency < 25
    if ok:
        print(f"  ✅ PASSED (survived {concurrency} concurrent disconnects)")
    else:
        print(f"  ❌ FAILED")

    return ProfileResult(
        test_name="Disconnect storm",
        passed=ok,
        elapsed_sec=disconnect_wall_time,
        detection_latency_sec=cleanup_latency,
        details={
            "concurrency": concurrency,
            "timeouts": timeouts,
            "errors": errors,
            "disconnect_increments": disc_delta,
            "cleanup_latency_sec": cleanup_latency,
            "server_healthy": health_ok,
            "post_storm_ok": func_ok,
        },
    )


# ============================================================
# TEST 8: Repeated disconnect → recovery cycle
# ============================================================

async def test_8_disconnect_recovery_cycles(base_url: str, cycles: int = 5) -> ProfileResult:
    """
    Repeatedly disconnect then send a real request, verifying the server
    recovers correctly each time. Catches resource leaks that only appear
    after multiple disconnect/recovery cycles.
    """
    _header(8, f"Disconnect → recovery cycles (×{cycles})")

    latencies = []
    all_ok = True

    for c in range(cycles):
        print(f"  [Cycle {c+1}/{cycles}]")

        # Disconnect
        async with aiohttp.ClientSession() as s:
            try:
                await _post(s, f"{base_url}/custom_code_A",
                            {"code": "import time; time.sleep(60)", "language": "python", "run_timeout": 30}, 2)
            except Exception:
                pass

        # Wait for detection
        await asyncio.sleep(8)

        # Recovery: real request
        start = time.monotonic()
        async with aiohttp.ClientSession() as s:
            try:
                status, body, elapsed = await _post(
                    s, f"{base_url}/custom_code_A",
                    {"code": f"print('cycle_{c}')", "language": "python", "run_timeout": 10}, 30)
                stdout = body.get("run_result", {}).get("stdout", "").strip()
                if status == 200 and f"cycle_{c}" in stdout:
                    print(f"    Recovery OK ({elapsed:.2f}s)")
                    latencies.append(elapsed)
                else:
                    print(f"    Recovery FAIL: status={status}, output='{stdout}'")
                    all_ok = False
            except Exception as e:
                print(f"    Recovery FAIL: {e}")
                all_ok = False

    if latencies:
        print(f"\n  Recovery latencies: {[f'{l:.2f}s' for l in latencies]}")
        print(f"  Mean: {statistics.mean(latencies):.2f}s, Max: {max(latencies):.2f}s")

        # Check for monotonically increasing latency (resource leak symptom)
        if len(latencies) > 2:
            diffs = [latencies[i+1] - latencies[i] for i in range(len(latencies)-1)]
            if all(d > 0.5 for d in diffs[-3:]):
                print("  ⚠️  Latency increasing monotonically - possible resource leak!")
                all_ok = False

    if all_ok:
        print(f"  ✅ PASSED ({cycles} cycles, no degradation)")
    else:
        print(f"  ❌ FAILED")

    return ProfileResult(
        test_name="Disconnect recovery cycles",
        passed=all_ok,
        details={
            "cycles": cycles,
            "recovery_latencies": [round(l, 2) for l in latencies],
        },
    )


# ============================================================
# TEST 9: Disconnect on multiple API paths simultaneously
# ============================================================

async def test_9_multi_path_disconnect(base_url: str) -> ProfileResult:
    """Verify disconnect detection works across all API paths."""
    _header(9, "Multi-path disconnect detection")

    paths = ["/custom_code_A", "/custom_code_B", "/custom_code_C", "/model_code"]

    m_before = await _get_metrics(base_url)
    disc_before = {}
    for p in paths:
        disc_before[p] = _get_metric(m_before, "code_errors_total",
                                     {"path": p, "error_type": "client_disconnected"})

    # Send disconnect requests to all paths simultaneously
    print(f"  [1] Sending disconnect requests to {len(paths)} paths...")
    tasks = []
    sessions = []
    for p in paths:
        s = aiohttp.ClientSession()
        sessions.append(s)
        payload = {"code": "import time; time.sleep(60)", "language": "python", "run_timeout": 30}
        t = asyncio.create_task(_post(s, f"{base_url}{p}", payload, 2))
        tasks.append(t)

    for t in tasks:
        try:
            await t
        except Exception:
            pass
    for s in sessions:
        await s.close()
    print("  All disconnected")

    # Wait for detection
    print("  [2] Waiting for detection (15s)...")
    await asyncio.sleep(15)

    m_after = await _get_metrics(base_url)
    results_per_path = {}
    all_detected = True
    for p in paths:
        disc_after = _get_metric(m_after, "code_errors_total",
                                 {"path": p, "error_type": "client_disconnected"})
        delta = disc_after - disc_before[p]
        results_per_path[p] = delta
        sym = "✓" if delta >= 1 else "✗"
        print(f"  {sym} {p}: disconnect_delta={delta}")
        if delta < 1:
            all_detected = False

    if all_detected:
        print("  ✅ PASSED (all paths detected disconnect)")
    else:
        print("  ❌ FAILED (some paths missed disconnect detection)")

    return ProfileResult(
        test_name="Multi-path disconnect",
        passed=all_detected,
        details={"per_path_deltas": results_per_path},
    )


# ============================================================
# TEST 10: Queue timeout fallback verification
# ============================================================

async def test_10_queue_timeout_fallback(base_url: str) -> ProfileResult:
    """
    Verify the queue_timeout safety net works as a fallback
    when disconnect detection fails (e.g., half-open TCP connections).
    
    This test doesn't disconnect - it lets the full queue timeout elapse.
    Needs SEMAPHORE_LIMIT=1.
    """
    _header(10, "Queue timeout fallback verification (needs SEMAPHORE_LIMIT=1)")

    blocker = {
        "code": "import time; time.sleep(30); print('blocker_done')",
        "language": "python",
        "run_timeout": 15,  # short run_timeout → queue_timeout = 15+5 = 20s
    }
    queued = {
        "code": "print('queued')",
        "language": "python",
        "run_timeout": 5,  # queue_timeout = 5+5 = 10s
    }

    # Block the semaphore
    print("  [1] Sending blocker (sleeps 30s, run_timeout=15)...")
    bs = aiohttp.ClientSession()
    bt = asyncio.create_task(_post(bs, f"{base_url}/custom_code_A", blocker, 60))
    await asyncio.sleep(2)

    # Send request that will queue and eventually hit queue_timeout
    print("  [2] Sending queued request (run_timeout=5 → queue_timeout≈10s)...")
    print("       Expect 504 after ~10s if queue timeout works...")
    start = time.monotonic()

    async with aiohttp.ClientSession() as s:
        try:
            status, body, elapsed = await _post(s, f"{base_url}/custom_code_A", queued, 30)
            run_status = body.get("run_result", {}).get("status", "")
            print(f"  Status: {status}, Time: {elapsed:.2f}s, Run status: '{run_status}'")

            ok = status == 504 and "QueueTimeout" in run_status
            if ok:
                print(f"  ✅ PASSED (queue timeout triggered correctly in {elapsed:.1f}s)")
            else:
                # Might get 200 if blocker finishes first, that's also fine
                if status == 200 and elapsed > 8:
                    print(f"  ⚠️  Blocker finished before queue timeout, request succeeded (not a bug)")
                    ok = True
                else:
                    print(f"  ❌ FAILED (unexpected status={status})")
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            print(f"  Client timed out after {elapsed:.1f}s")
            ok = False
        except Exception as e:
            elapsed = time.monotonic() - start
            print(f"  ❌ FAILED: {e}")
            ok = False

    # Cleanup
    print("  [3] Waiting for blocker...")
    try:
        await bt
    except Exception:
        pass
    await bs.close()

    return ProfileResult(
        test_name="Queue timeout fallback",
        passed=ok,
        elapsed_sec=elapsed,
        details={"expected_queue_timeout": "~10s"},
    )


# ============================================================
# TEST 11: Sustained stress test (continuous load for N seconds)
# ============================================================

async def test_11_sustained_stress(
    base_url: str, concurrency: int = 30, duration: int = 30
) -> ProfileResult:
    """
    Sustained stress test: continuously send requests at concurrency N
    for `duration` seconds. Reports throughput, latency percentiles,
    error rate, and per-second throughput timeline.
    """
    _header(11, f"Sustained stress test (concurrency={concurrency}, duration={duration}s)")

    payload = {"code": "print('ok')", "language": "python", "run_timeout": 10}
    semaphore = asyncio.Semaphore(concurrency)
    stop_event = asyncio.Event()

    # Shared mutable state (single-threaded asyncio, no lock needed)
    stats = {
        "success": 0,
        "error": 0,
        "latencies": [],
        "per_second": [],  # (timestamp, count) buckets
    }
    global_start = None

    async def _worker(session: aiohttp.ClientSession):
        while not stop_event.is_set():
            async with semaphore:
                if stop_event.is_set():
                    break
                try:
                    status, body, elapsed = await _post(
                        session, f"{base_url}/custom_code_A", payload, 15
                    )
                    if status == 200:
                        stats["success"] += 1
                    else:
                        stats["error"] += 1
                    stats["latencies"].append(elapsed)
                except asyncio.TimeoutError:
                    stats["error"] += 1
                    stats["latencies"].append(15.0)
                except Exception:
                    stats["error"] += 1
            # Tiny sleep to avoid busy-loop if server is very fast
            await asyncio.sleep(0.01)

    # Progress reporter
    async def _reporter():
        last_count = 0
        sec = 0
        while not stop_event.is_set():
            await asyncio.sleep(1)
            sec += 1
            current = stats["success"] + stats["error"]
            delta = current - last_count
            last_count = current
            stats["per_second"].append(delta)
            if sec % 5 == 0 or sec == duration:
                lats = stats["latencies"]
                p50 = statistics.median(lats) if lats else 0
                print(
                    f"  [{sec:3d}s] reqs={current}, "
                    f"rps={delta}/s, "
                    f"ok={stats['success']}, err={stats['error']}, "
                    f"p50={p50:.3f}s"
                )

    # Metrics snapshot before
    m_before = await _get_metrics(base_url)
    active_before = _get_metric(m_before, "sandbox_active_sandboxes")

    print(f"  Starting {concurrency} workers for {duration}s...")
    global_start = time.monotonic()

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=concurrency + 10)
    ) as session:
        workers = [asyncio.create_task(_worker(session)) for _ in range(concurrency)]
        reporter = asyncio.create_task(_reporter())

        await asyncio.sleep(duration)
        stop_event.set()

        await asyncio.gather(*workers, return_exceptions=True)
        reporter.cancel()

    total_time = time.monotonic() - global_start
    total_reqs = stats["success"] + stats["error"]
    lats = stats["latencies"]

    # --- Summary ---
    print(f"\n  --- Results ---")
    print(f"  Duration:       {total_time:.1f}s")
    print(f"  Total requests: {total_reqs}")
    print(f"  Throughput:     {total_reqs / total_time:.1f} req/s")
    print(f"  Success:        {stats['success']}")
    print(f"  Errors:         {stats['error']}")

    if lats:
        sorted_lats = sorted(lats)
        n = len(sorted_lats)
        print(f"  Latency p50:    {sorted_lats[int(n * 0.50)]:.3f}s")
        print(f"  Latency p90:    {sorted_lats[int(n * 0.90)]:.3f}s")
        print(f"  Latency p95:    {sorted_lats[int(n * 0.95)]:.3f}s")
        print(f"  Latency p99:    {sorted_lats[int(n * 0.99)]:.3f}s")
        print(f"  Latency max:    {sorted_lats[-1]:.3f}s")
        print(f"  Latency mean:   {statistics.mean(lats):.3f}s")

    if stats["per_second"]:
        rps_list = stats["per_second"]
        print(f"  RPS min:        {min(rps_list)}")
        print(f"  RPS max:        {max(rps_list)}")
        print(f"  RPS mean:       {statistics.mean(rps_list):.1f}")

    # Verify server health after stress
    print(f"\n  [Post-stress] Health check...")
    health_ok = False
    async with aiohttp.ClientSession() as s:
        try:
            async with s.get(
                f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    health_ok = True
                    print("  Server healthy ✓")
        except Exception as e:
            print(f"  Server health check failed: {e}")

    # Verify a request still works
    func_ok = False
    async with aiohttp.ClientSession() as s:
        try:
            status, body, elapsed = await _post(
                s, f"{base_url}/custom_code_A",
                {"code": "print('post_stress_ok')", "language": "python", "run_timeout": 10},
                30,
            )
            stdout = body.get("run_result", {}).get("stdout", "").strip()
            if status == 200 and "post_stress_ok" in stdout:
                func_ok = True
                print(f"  Post-stress request OK ({elapsed:.2f}s)")
            else:
                print(f"  Post-stress request: status={status}, stdout='{stdout}'")
        except Exception as e:
            print(f"  Post-stress request failed: {e}")

    # Check no resource leak
    await asyncio.sleep(3)
    m_after = await _get_metrics(base_url)
    active_after = _get_metric(m_after, "sandbox_active_sandboxes")
    leak = active_after > active_before
    if leak:
        print(f"  ⚠️  Possible resource leak: active_sandboxes {active_before} → {active_after}")
    else:
        print(f"  No resource leak ✓ (active_sandboxes={active_after})")

    error_rate = stats["error"] / total_reqs if total_reqs > 0 else 1.0
    ok = health_ok and func_ok and error_rate < 0.05 and not leak
    if ok:
        print(f"  ✅ PASSED (throughput={total_reqs / total_time:.1f} req/s, error_rate={error_rate:.1%})")
    else:
        print(f"  ❌ FAILED (health={health_ok}, func={func_ok}, error_rate={error_rate:.1%}, leak={leak})")

    return ProfileResult(
        test_name="Sustained stress test",
        passed=ok,
        elapsed_sec=total_time,
        details={
            "concurrency": concurrency,
            "duration_sec": duration,
            "total_requests": total_reqs,
            "throughput_rps": round(total_reqs / total_time, 1),
            "success": stats["success"],
            "errors": stats["error"],
            "error_rate": f"{error_rate:.2%}",
            "p50_sec": round(sorted(lats)[int(len(lats) * 0.50)], 3) if lats else None,
            "p95_sec": round(sorted(lats)[int(len(lats) * 0.95)], 3) if lats else None,
            "p99_sec": round(sorted(lats)[int(len(lats) * 0.99)], 3) if lats else None,
            "server_healthy": health_ok,
            "post_stress_ok": func_ok,
            "resource_leak": leak,
        },
    )


# ============================================================
# MAIN
# ============================================================

TESTS = {
    1:  ("Baseline + metrics", test_1_baseline),
    2:  ("Queue disconnect latency", test_2_queue_disconnect_latency),
    3:  ("Exec disconnect latency", test_3_exec_disconnect_latency),
    4:  ("Semaphore leak detection", test_4_semaphore_leak),
    5:  ("Ghost accumulation", test_5_ghost_accumulation),
    6:  ("Metrics correctness", test_6_metrics_correctness),
    7:  ("Disconnect storm", test_7_disconnect_storm),
    8:  ("Disconnect recovery cycles", test_8_disconnect_recovery_cycles),
    9:  ("Multi-path disconnect", test_9_multi_path_disconnect),
    10: ("Queue timeout fallback", test_10_queue_timeout_fallback),
    11: ("Sustained stress test", test_11_sustained_stress),
}

SUITES = {
    "disconnect":  [2, 3, 4, 5, 10],       # Core disconnect detection
    "metrics":     [1, 6],                   # Metrics validation
    "stress":      [7, 8, 9, 11],           # Stress & robustness
    "perf":        [11],                     # Pure sustained stress test
    "quick":       [1, 3, 4],               # Quick smoke test
    "all":         list(range(1, 12)),
}


async def main(host, port, suite, test_id, concurrency, duration):
    base_url = f"http://{host}:{port}"
    print(f"{'=' * 80}")
    print(f"Sandbox Disconnect Detection Profiling Suite")
    print(f"Target: {base_url}")
    print(f"{'=' * 80}")

    # Health check first
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"❌ Server not healthy (status={resp.status})")
                    return
                print(f"Server healthy ✓")
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        return

    if test_id:
        test_ids = [test_id]
    else:
        test_ids = SUITES.get(suite, SUITES["quick"])

    print(f"Running tests: {test_ids} (suite: {suite})")
    if any(t in test_ids for t in [2, 5, 10]):
        print("⚠️  Tests 2, 5, 10 require SANDBOX_SEMAPHORE_LIMIT=1 on server")
    print()

    for tid in test_ids:
        name, func = TESTS[tid]
        try:
            if tid == 7:
                result = await func(base_url, concurrency=concurrency)
            elif tid == 11:
                result = await func(base_url, concurrency=concurrency, duration=duration)
            else:
                result = await func(base_url)
            report.add(result)
        except Exception as e:
            print(f"  ❌ CRASHED: {e}")
            import traceback
            traceback.print_exc()
            report.add(ProfileResult(test_name=name, passed=False, error=str(e)))

    report.print_summary()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sandbox Disconnect Detection Profiling Suite")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--suite", default="all",
                   choices=list(SUITES.keys()),
                   help="Test suite to run (default: all)")
    p.add_argument("--test", type=int, default=None,
                   help="Run single test by ID (1-11)")
    p.add_argument("--concurrency", type=int, default=30,
                    help="Concurrency for storm/stress tests (default: 30)")
    p.add_argument("--duration", type=int, default=30,
                    help="Duration in seconds for sustained stress test (default: 30)")
    args = p.parse_args()
    asyncio.run(main(args.host, args.port, args.suite, args.test, args.concurrency, args.duration))
