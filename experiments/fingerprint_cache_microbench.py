"""Microbench: Python OrderedDict LRU vs Rust engine fingerprint cache.

Phase 9 bench gate: no regression vs the OrderedDict baseline. PyO3
crossing cost is real, so a 2-3x slower-per-call is acceptable so long
as throughput at realistic batch sizes stays well inside ms-scale.
"""

import tempfile
import time
from collections import OrderedDict

import tardigrade_db

CAPACITY = 256
ITERS = 100_000


def bench_python(iters):
    cache: OrderedDict[int, int] = OrderedDict()
    start = time.perf_counter()
    for i in range(iters):
        fp = i & 0x3FF  # 1024 distinct fingerprints — exercises eviction
        cache[fp] = i
        cache.move_to_end(fp)
        while len(cache) > CAPACITY:
            cache.popitem(last=False)
        if i % 3 == 0:
            cache.get(fp ^ 0x55, None)
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1e9


def bench_rust(iters):
    with tempfile.TemporaryDirectory() as tmp:
        eng = tardigrade_db.Engine(tmp)
        eng.set_fingerprint_capacity(CAPACITY)
        start = time.perf_counter()
        for i in range(iters):
            fp = i & 0x3FF
            eng.fingerprint_put(fp, i)
            if i % 3 == 0:
                eng.fingerprint_get(fp ^ 0x55)
        elapsed = time.perf_counter() - start
        return elapsed / iters * 1e9


def main():
    # Warmup
    bench_python(1000)
    bench_rust(1000)
    py_ns = bench_python(ITERS)
    rs_ns = bench_rust(ITERS)
    print(f"capacity={CAPACITY}, iters={ITERS}")
    print(f"  python (OrderedDict)       {py_ns:6.0f} ns/call")
    print(f"  rust (engine.fingerprint)  {rs_ns:6.0f} ns/call")
    print(f"  ratio: {rs_ns / py_ns:.2f}x ({'slower' if rs_ns > py_ns else 'faster'})")


if __name__ == "__main__":
    main()
