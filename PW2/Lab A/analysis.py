"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# TODO 1: read freefall.csv into arrays t and y
#         (hint: np.loadtxt with a comma delimiter, skipping the header)
data = np.loadtxt("freefall.csv", delimiter=",", skiprows=1)

t = data[:, 0]
y = data[:, 1]

# TODO 2: compute velocity v = derivative of y w.r.t. t   (np.gradient)
#         and acceleration a = derivative of v w.r.t. t    (np.gradient again)
#         Print the mean acceleration. Is it close to -9.81? Is it noisy?

v = np.gradient(y, t)
a = np.gradient(v, t)

print("Mean acceleration:", np.mean(a))
print("Standard deviation:", np.std(a))

# TODO 3: integrate a back up to recover velocity and position
#         (hint: cumulative_trapezoid(a, t, initial=0) + v[0], then again)

v_recovered = cumulative_trapezoid(a, t, initial=0) + v[0]
y_recovered = cumulative_trapezoid(v_recovered, t, initial=0) + y[0]

# TODO 4: make a figure with 3 stacked panels: position, velocity, acceleration
#         vs time. Mark the true -9.81 line on the acceleration panel.
#         Save it as motion.png


fig, axes = plt.subplots(3, 1, figsize=(8, 10))

axes[0].plot(t, y, label="Measured position")
axes[0].plot(t, y_recovered, label="Recovered position")
axes[0].set_ylabel("Position")
axes[0].legend()

axes[1].plot(t, v, label="Measured velocity")
axes[1].plot(t, v_recovered, label="Recovered velocity")
axes[1].set_ylabel("Velocity")
axes[1].legend()

axes[2].plot(t, a, label="Acceleration")
axes[2].axhline(-9.81, color="red", linestyle="--", label="True -9.81")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Acceleration")
axes[2].legend()

plt.tight_layout()
plt.savefig("motion.png")

