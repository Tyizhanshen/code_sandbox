#!/usr/bin/env python3
"""
Sandbox Performance Stress Test Script

Tests 5 scenarios:
1. Fast request throughput (QPS measurement)
2. Mixed load (fast + slow/infinite loop)
3. Cascade recovery (all timeout → fast requests)
4. Client disconnect (early abort)
5. Adaptive semaphore observation

Usage:
    python test_sandbox_load.py --url http://10.103.172.50:8080 --scenario all
    python test_sandbox_load.py --url http://10.103.172.50:8080 --scenario fast_throughput --concurrency 500
    python test_sandbox_load.py --url http://10.103.172.50:8080 --scenario mixed_load
    python test_sandbox_load.py --url http://10.103.172.50:8080 --scenario cascade_recovery
    python test_sandbox_load.py --url http://10.103.172.50:8080 --scenario client_disconnect
    python test_sandbox_load.py --url http://10.103.172.50:8080 --scenario adaptive_semaphore
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Test Code Snippets
# ============================================================
FAST_CODE = 'print("hello world")'
SLOW_CODE_TEMPLATE = "import time; time.sleep({duration})"
INFINITE_LOOP_CODE = "while True: pass"
CPU_HEAVY_CODE = """
total = 0
for i in range(10**7):
    total += i * i
print(total)
"""


@dataclass
class RequestResult:
    """Result of a single sandbox request."""
    status: str = "unknown"
    run_status: str = "unknown"
    http_status: int = 0
    latency: float = 0.0
    error: Optional[str] = None


@dataclass
class ScenarioStats:
    """Aggregated statistics for a test scenario."""
    scenario_name: str = ""
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    timeout: int = 0
    queue_timeout: int = 0
    client_disconnect: int = 0
    http_errors: int = 0
    latencies: list = field(default_factory=list)
    status_distribution: Counter = field(default_factory=Counter)
    duration: float = 0.0

    def add(self, result: RequestResult):
        self.total_requests += 1
        self.latencies.append(result.latency)
        self.status_distribution[result.status] += 1

        if result.status == "Success":
            self.successful += 1
        elif result.status == "Failed" and result.run_status == "TimeLimitExceeded":
            self.timeout += 1
        elif result.status == "SandboxError" and result.run_status == "QueueTimeout":
            self.queue_timeout += 1
        elif result.status == "Aborted":
            self.client_disconnect += 1
        elif result.http_status >= 400 or result.error:
            self.http_errors += 1
        else:
            self.failed += 1

    def report(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  Scenario: {self.scenario_name}",
            f"{'='*60}",
            f"  Total requests:    {self.total_requests}",
            f"  Duration:          {self.duration:.1f}s",
            f"  Throughput:        {self.total_requests / self.duration:.1f} req/s" if self.duration > 0 else "",
            f"",
            f"  ✅ Successful:     {self.successful}",
            f"  ❌ Failed:         {self.failed}",
            f"  ⏱  Timeout:        {self.timeout}",
            f"  🔒 Queue Timeout:  {self.queue_timeout}",
            f"  🔌 Client Discon:  {self.client_disconnect}",
            f"  🌐 HTTP Errors:    {self.http_errors}",
        ]

        if self.latencies:
            lines.extend([
                f"",
                f"  Latency (seconds):",
                f"    Min:    {min(self.latencies):.3f}",
                f"    Median: {statistics.median(self.latencies):.3f}",
                f"    P95:    {sorted(self.latencies)[int(len(self.latencies) * 0.95)]:.3f}",
                f"    P99:    {sorted(self.latencies)[int(len(self.latencies) * 0.99)]:.3f}",
                f"    Max:    {max(self.latencies):.3f}",
                f"    Mean:   {statistics.mean(self.latencies):.3f}",
            ])

        if self.status_distribution:
            lines.extend([
                f"",
                f"  Status Distribution:",
            ])
            for status, count in self.status_distribution.most_common():
                lines.append(f"    {status}: {count}")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


# ============================================================
# Core Request Function
# ============================================================
async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    code: str,
    run_timeout: int = 10,
    compile_timeout: int = 10,
    memory_limit_mb: int = 128,
    client_timeout: Optional[float] = None,
    language: str = "python3",
) -> RequestResult:
    """Send a single code execution request to the sandbox."""
    payload = {
        "code": code,
        "language": language,
        "run_timeout": run_timeout,
        "compile_timeout": compile_timeout,
        "stdin": "",
        "memory_limit_MB": memory_limit_mb,
        "files": {},
        "fetch_files": [],
        "truncate_output": True,
    }

    result = RequestResult()
    start = time.monotonic()

    try:
        timeout = aiohttp.ClientTimeout(total=client_timeout) if client_timeout else None
        async with session.post(url, json=payload, timeout=timeout) as resp:
            result.http_status = resp.status
            body = await resp.json()
            result.status = body.get("status", "unknown")
            run_result = body.get("run_result", {})
            result.run_status = run_result.get("status", "unknown") if run_result else "unknown"
    except asyncio.TimeoutError:
        result.status = "ClientTimeout"
        result.error = "Client-side timeout"
    except aiohttp.ClientError as e:
        result.status = "ClientError"
        result.error = str(e)
    except Exception as e:
        result.status = "UnexpectedError"
        result.error = str(e)

    result.latency = time.monotonic() - start
    return result


# ============================================================
# Test Scenarios
# ============================================================

async def scenario_fast_throughput(url: str, concurrency: int = 500, total: int = 1000) -> ScenarioStats:
    """
    Scenario 1: Fast request throughput test.
    Sends `total` fast requests with `concurrency` parallelism.
    Measures QPS and latency distribution.
    """
    stats = ScenarioStats(scenario_name=f"Fast Throughput (concurrency={concurrency}, total={total})")
    semaphore = asyncio.Semaphore(concurrency)

    async def _task(session):
        async with semaphore:
            return await send_request(
                session, url, FAST_CODE,
                run_timeout=10, client_timeout=30,
            )

    start = time.monotonic()
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_task(session) for _ in range(total)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    stats.duration = time.monotonic() - start

    for r in results:
        if isinstance(r, RequestResult):
            stats.add(r)
        else:
            stats.total_requests += 1
            stats.http_errors += 1

    return stats


async def scenario_mixed_load(
    url: str,
    concurrency: int = 200,
    fast_count: int = 100,
    slow_count: int = 200,
    slow_timeout: int = 100,
) -> ScenarioStats:
    """
    Scenario 2: Mixed fast + slow/infinite loop requests.
    Verifies that slow requests don't block fast requests.
    """
    stats = ScenarioStats(
        scenario_name=f"Mixed Load (fast={fast_count}, slow={slow_count}, concurrency={concurrency})"
    )
    fast_latencies = []
    slow_latencies = []

    async def _fast_task(session):
        r = await send_request(
            session, url, FAST_CODE,
            run_timeout=10, client_timeout=60,
        )
        fast_latencies.append(r.latency)
        return r

    async def _slow_task(session):
        r = await send_request(
            session, url, INFINITE_LOOP_CODE,
            run_timeout=slow_timeout, client_timeout=slow_timeout + 30,
        )
        slow_latencies.append(r.latency)
        return r

    start = time.monotonic()
    # Connection limit must be >= slow_count + fast_count to avoid TCP pool blocking
    total_conn = slow_count + fast_count
    connector = aiohttp.TCPConnector(limit=total_conn, limit_per_host=total_conn)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Phase 1: Launch all slow requests
        slow_tasks = [asyncio.create_task(_slow_task(session)) for _ in range(slow_count)]

        # Phase 2: Wait a bit, then launch fast requests independently
        await asyncio.sleep(2)
        logger.info("[MixedLoad] Slow requests running, now sending %d fast requests...", fast_count)
        fast_tasks = [asyncio.create_task(_fast_task(session)) for _ in range(fast_count)]

        # Wait for fast requests first (they should finish quickly if not blocked)
        fast_results = await asyncio.gather(*fast_tasks, return_exceptions=True)
        fast_done_time = time.monotonic() - start
        logger.info("[MixedLoad] Fast requests done in %.1fs", fast_done_time)

        # Wait for slow requests
        slow_results = await asyncio.gather(*slow_tasks, return_exceptions=True)

    stats.duration = time.monotonic() - start

    for r in [*fast_results, *slow_results]:
        if isinstance(r, RequestResult):
            stats.add(r)
        else:
            stats.total_requests += 1
            stats.http_errors += 1

    # Extra analysis: fast vs slow latency
    extra = []
    if fast_latencies:
        extra.append(f"  Fast requests latency: median={statistics.median(fast_latencies):.3f}s, "
                      f"max={max(fast_latencies):.3f}s")
    if slow_latencies:
        extra.append(f"  Slow requests latency: median={statistics.median(slow_latencies):.3f}s, "
                      f"max={max(slow_latencies):.3f}s")

    stats._extra = "\n".join(extra)
    return stats


async def scenario_cascade_recovery(
    url: str,
    timeout_count: int = 100,
    recovery_count: int = 50,
    run_timeout: int = 30,
) -> ScenarioStats:
    """
    Scenario 3: Cascade timeout recovery test.
    First floods server with timeout-inducing requests,
    then immediately sends fast requests to test recovery.
    """
    stats = ScenarioStats(
        scenario_name=f"Cascade Recovery (timeout={timeout_count}, recovery={recovery_count})"
    )
    concurrency = max(timeout_count, recovery_count)
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)

    start = time.monotonic()
    async with aiohttp.ClientSession(connector=connector) as session:
        # Phase 1: Flood with infinite loops
        logger.info("[Cascade] Phase 1: Sending %d infinite loop requests...", timeout_count)
        timeout_tasks = [
            send_request(
                session, url, INFINITE_LOOP_CODE,
                run_timeout=run_timeout, client_timeout=run_timeout + 30,
            )
            for _ in range(timeout_count)
        ]
        timeout_results = await asyncio.gather(*timeout_tasks, return_exceptions=True)
        phase1_end = time.monotonic()
        logger.info("[Cascade] Phase 1 done in %.1fs", phase1_end - start)

        # Phase 2: Immediately send fast requests
        logger.info("[Cascade] Phase 2: Sending %d fast requests for recovery test...", recovery_count)
        phase2_start = time.monotonic()
        recovery_tasks = [
            send_request(
                session, url, FAST_CODE,
                run_timeout=10, client_timeout=60,
            )
            for _ in range(recovery_count)
        ]
        recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
        phase2_end = time.monotonic()
        logger.info("[Cascade] Phase 2 done in %.1fs", phase2_end - phase2_start)

    stats.duration = time.monotonic() - start

    # Report Phase 1
    for r in timeout_results:
        if isinstance(r, RequestResult):
            stats.add(r)

    # Report Phase 2 (recovery)
    recovery_latencies = []
    recovery_success = 0
    for r in recovery_results:
        if isinstance(r, RequestResult):
            stats.add(r)
            recovery_latencies.append(r.latency)
            if r.status == "Success":
                recovery_success += 1

    if recovery_latencies:
        logger.info(
            "[Cascade] Recovery: %d/%d succeeded, median latency=%.3fs, max=%.3fs",
            recovery_success, recovery_count,
            statistics.median(recovery_latencies),
            max(recovery_latencies),
        )

    return stats


async def scenario_client_disconnect(
    url: str,
    count: int = 50,
    disconnect_after: float = 5.0,
) -> ScenarioStats:
    """
    Scenario 4: Client disconnect test.
    Sends requests then disconnects after `disconnect_after` seconds.
    Verifies server detects disconnect and reclaims resources.
    """
    stats = ScenarioStats(
        scenario_name=f"Client Disconnect (count={count}, disconnect_after={disconnect_after}s)"
    )

    connector = aiohttp.TCPConnector(limit=count, limit_per_host=count)

    start = time.monotonic()
    async with aiohttp.ClientSession(connector=connector) as session:
        # Send requests with very short client timeout (simulate disconnect)
        tasks = [
            send_request(
                session, url, SLOW_CODE_TEMPLATE.format(duration=60),
                run_timeout=120, client_timeout=disconnect_after,
            )
            for _ in range(count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    stats.duration = time.monotonic() - start

    for r in results:
        if isinstance(r, RequestResult):
            stats.add(r)
            if r.status == "ClientTimeout":
                stats.client_disconnect += 1
        else:
            stats.total_requests += 1
            stats.http_errors += 1

    # Verify server recovered: send fast requests after disconnect
    logger.info("[Disconnect] Waiting 10s for server to detect disconnects...")
    await asyncio.sleep(10)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as session:
        verify_tasks = [
            send_request(
                session, url, FAST_CODE,
                run_timeout=10, client_timeout=30,
            )
            for _ in range(20)
        ]
        verify_results = await asyncio.gather(*verify_tasks, return_exceptions=True)

    verify_success = sum(
        1 for r in verify_results
        if isinstance(r, RequestResult) and r.status == "Success"
    )
    logger.info(
        "[Disconnect] Post-disconnect verification: %d/%d fast requests succeeded",
        verify_success, len(verify_results),
    )

    return stats


async def scenario_adaptive_semaphore(
    url: str,
    metrics_url: Optional[str] = None,
    duration: int = 120,
) -> ScenarioStats:
    """
    Scenario 5: Observe adaptive semaphore under varying load.
    Alternates between high and low load, monitoring semaphore metrics.
    """
    stats = ScenarioStats(
        scenario_name=f"Adaptive Semaphore Observation (duration={duration}s)"
    )

    if not metrics_url:
        # Derive metrics URL from sandbox URL
        base = url.rsplit("/", 1)[0] if "/" in url else url
        metrics_url = f"{base}/metrics"

    connector = aiohttp.TCPConnector(limit=300, limit_per_host=300)
    interval = duration // 4  # 4 phases

    async def _fetch_semaphore_limit(session):
        try:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                text = await resp.text()
                for line in text.split("\n"):
                    if line.startswith("sandbox_semaphore_current_limit"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[-1])
        except Exception:
            pass
        return None

    start = time.monotonic()
    async with aiohttp.ClientSession(connector=connector) as session:
        # Phase 1: High load (infinite loops)
        logger.info("[Adaptive] Phase 1: High CPU load (%ds)...", interval)
        high_tasks = []
        for _ in range(200):
            high_tasks.append(
                send_request(
                    session, url, INFINITE_LOOP_CODE,
                    run_timeout=interval + 10,
                    client_timeout=interval + 20,
                )
            )

        # Monitor semaphore during high load
        for _ in range(interval // 5):
            await asyncio.sleep(5)
            limit = await _fetch_semaphore_limit(session)
            elapsed = time.monotonic() - start
            if limit is not None:
                logger.info("[Adaptive] t=%.0fs semaphore_limit=%.0f", elapsed, limit)

        # Phase 2: Let load die down (wait for timeouts)
        logger.info("[Adaptive] Phase 2: Waiting for load to decrease (%ds)...", interval)
        for _ in range(interval // 5):
            await asyncio.sleep(5)
            limit = await _fetch_semaphore_limit(session)
            elapsed = time.monotonic() - start
            if limit is not None:
                logger.info("[Adaptive] t=%.0fs semaphore_limit=%.0f", elapsed, limit)

        # Collect results
        results = await asyncio.gather(*high_tasks, return_exceptions=True)

        # Phase 3: Low load → semaphore should grow back
        logger.info("[Adaptive] Phase 3: Low load, semaphore should recover (%ds)...", interval)
        low_tasks = []
        for _ in range(20):
            low_tasks.append(
                send_request(
                    session, url, FAST_CODE,
                    run_timeout=10, client_timeout=30,
                )
            )
        low_results = await asyncio.gather(*low_tasks, return_exceptions=True)

        for _ in range(interval // 5):
            await asyncio.sleep(5)
            limit = await _fetch_semaphore_limit(session)
            elapsed = time.monotonic() - start
            if limit is not None:
                logger.info("[Adaptive] t=%.0fs semaphore_limit=%.0f", elapsed, limit)

    stats.duration = time.monotonic() - start

    for r in [*results, *low_results]:
        if isinstance(r, RequestResult):
            stats.add(r)

    return stats


# ============================================================
# Main
# ============================================================

SCENARIO_MAP = {
    "fast_throughput": scenario_fast_throughput,
    "mixed_load": scenario_mixed_load,
    "cascade_recovery": scenario_cascade_recovery,
    "client_disconnect": scenario_client_disconnect,
    "adaptive_semaphore": scenario_adaptive_semaphore,
}


async def run_scenario(name: str, url: str, **kwargs):
    """Run a single scenario and print results."""
    endpoint = f"{url}/custom_code"
    fn = SCENARIO_MAP[name]
    logger.info("Starting scenario: %s", name)
    stats = await fn(endpoint, **kwargs)
    print(stats.report())
    if hasattr(stats, '_extra'):
        print(stats._extra)
    return stats


async def run_all(url: str, concurrency: int):
    """Run all scenarios sequentially."""
    all_stats = []

    logger.info("=" * 60)
    logger.info("  Running ALL scenarios against %s", url)
    logger.info("=" * 60)

    # 1. Fast throughput
    s = await run_scenario("fast_throughput", url, concurrency=concurrency, total=concurrency)
    all_stats.append(s)

    await asyncio.sleep(5)

    # 2. Mixed load
    s = await run_scenario("mixed_load", url, concurrency=min(concurrency, 200))
    all_stats.append(s)

    await asyncio.sleep(5)

    # 3. Cascade recovery
    s = await run_scenario("cascade_recovery", url, timeout_count=100, recovery_count=50)
    all_stats.append(s)

    await asyncio.sleep(5)

    # 4. Client disconnect
    s = await run_scenario("client_disconnect", url, count=50)
    all_stats.append(s)

    await asyncio.sleep(5)

    # 5. Adaptive semaphore
    s = await run_scenario("adaptive_semaphore", url, duration=120)
    all_stats.append(s)

    # Summary
    print("\n" + "=" * 60)
    print("  OVERALL SUMMARY")
    print("=" * 60)
    for s in all_stats:
        total = s.total_requests
        success_rate = (s.successful / total * 100) if total > 0 else 0
        print(f"  {s.scenario_name}")
        print(f"    Success rate: {success_rate:.1f}%  |  QPS: {total / s.duration:.1f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Sandbox Load Test")
    parser.add_argument("--url", required=True, help="Sandbox server base URL (e.g., http://10.103.172.50:8080)")
    parser.add_argument("--scenario", default="all",
                        choices=["all"] + list(SCENARIO_MAP.keys()),
                        help="Which scenario to run")
    parser.add_argument("--concurrency", type=int, default=500,
                        help="Concurrency level for fast_throughput (default: 500)")
    parser.add_argument("--duration", type=int, default=120,
                        help="Duration for adaptive_semaphore scenario (default: 120)")
    args = parser.parse_args()

    if args.scenario == "all":
        asyncio.run(run_all(args.url, args.concurrency))
    elif args.scenario == "fast_throughput":
        asyncio.run(run_scenario(args.scenario, args.url,
                                  concurrency=args.concurrency, total=args.concurrency))
    elif args.scenario == "adaptive_semaphore":
        asyncio.run(run_scenario(args.scenario, args.url, duration=args.duration))
    else:
        asyncio.run(run_scenario(args.scenario, args.url))


if __name__ == "__main__":
    main()
