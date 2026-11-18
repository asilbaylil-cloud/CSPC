import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the seismic data
d = pd.read_csv("PW5/data/seismogram_A.csv")

t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()
N = d["N"].to_numpy()
E = d["E"].to_numpy()

# calculate the sampling frequency
fs = 1 / (t[1] - t[0])
print("Sampling frequency:", fs, "Hz")

# calculate the FFT of Z
X = np.fft.fft(Z)
freq = np.fft.fftfreq(len(Z), d=1/fs)

amplitude = np.abs(X) / len(Z) * 2

# plot amplitude against index
plt.figure(figsize=(8, 4))
plt.plot(amplitude)
plt.xlabel("Index")
plt.ylabel("Amplitude")
plt.title("FFT of Z channel")
plt.savefig("PW5/spectrum_index.png")
plt.show()

# plot amplitude against frequency
plt.figure(figsize=(8, 4))
plt.plot(freq, amplitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of Z channel")
plt.savefig("PW5/spectrum_frequency.png")
plt.show()

# compare quiet part and whole record
quiet_Z = Z[t < 60]

X_quiet = np.fft.fft(quiet_Z)
f_quiet = np.fft.fftfreq(len(quiet_Z), d=1/fs)

amp_quiet = np.abs(X_quiet) / len(quiet_Z) * 2

# positive frequencies only
mask_all = freq > 0
mask_quiet = f_quiet > 0

plt.figure(figsize=(8, 5))
plt.loglog(freq[mask_all], amplitude[mask_all], label="Whole record")
plt.loglog(f_quiet[mask_quiet], amp_quiet[mask_quiet], label="Quiet part")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Quiet part vs Whole record")
plt.legend()
plt.savefig("PW5/spectrum_comparison.png")
plt.show()

# plot three channels
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

axes[0].plot(t, Z * 1000)
axes[0].set_ylabel("Z (mm/s)")

axes[1].plot(t, N * 1000)
axes[1].set_ylabel("N (mm/s)")

axes[2].plot(t, E * 1000)
axes[2].set_ylabel("E (mm/s)")
axes[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("PW5/record.png")
plt.show()
# check the inverse Fourier transform
Z_back = np.fft.ifft(np.fft.fft(Z)).real

print("Inverse FFT check:", np.allclose(Z_back, Z))
# band-pass filter
def bandpass(x, fs, a, b):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), d=1/fs)

    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0

    clean = np.fft.ifft(X).real
    return clean
Z_clean = bandpass(Z, fs, 0.5, 10)
N_clean = bandpass(N, fs, 0.5, 10)
E_clean = bandpass(E, fs, 0.5, 10)

# plot removed signal
removed = Z - Z_clean

plt.figure(figsize=(10, 4))
plt.plot(t, removed)
plt.xlabel("Time (s)")
plt.ylabel("Z (mm/s)")
plt.title("Removed signal")
plt.tight_layout()
plt.savefig("PW5/removed_signal.png")
plt.show()