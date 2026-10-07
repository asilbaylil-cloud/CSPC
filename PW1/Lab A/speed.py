import time
from decay import simulate_loop, simulate

N0 = 200000
lam = 0.4

start = time.perf_counter()
simulate_loop(N0, lam)
loop_time = time.perf_counter() - start

start = time.perf_counter()
simulate(N0, lam)
numpy_time = time.perf_counter() - start

speedup = loop_time / numpy_time

print(f"Loop time: {loop_time:.4f} s")
print(f"NumPy time: {numpy_time:.4f} s")
print(f"NumPy is {speedup:.2f}x faster")
