import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

# envelope
def envelope(x, window_seconds):
    w = int(window_seconds * fs)
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")


# z channel with three different windows
env_Z_005 = envelope(Z_clean, 0.05)
env_Z_05 = envelope(Z_clean, 0.5)
env_Z_5 = envelope(Z_clean, 5)

# horizontal motion
horizontal = np.sqrt(N_clean**2 + E_clean**2)
env_horizontal = envelope(horizontal, 0.5)

# compare the three windows
plt.figure(figsize=(10, 5))
plt.plot(t, env_Z_005, label="0.05 s")
plt.plot(t, env_Z_05, label="0.5 s")
plt.plot(t, env_Z_5, label="5 s")
plt.plot(t, env_horizontal, label="Horizontal motion (0.5 s)")

plt.xlabel("Time (s)")
plt.ylabel("Envelope")
plt.title("Z channel envelope")
plt.legend()
plt.tight_layout()
plt.savefig("PW5/envelopes.png")
plt.show() 

# arrival times

from scipy.signal import find_peaks

# pwave arrival: 5% of the maximum Z envelope
p_limit = 0.05 * np.max(env_Z_05)
p_indices = np.where(env_Z_05 > p_limit)[0]
tp_index = p_indices[0]
tp = t[tp_index]

# swave arrival: 10% of the maximum horizontal envelope,
# but only after tp + 2 seconds
s_limit = 0.10 * np.max(env_horizontal)
s_indices = np.where((t > tp + 2) & (env_horizontal > s_limit))[0]
ts_index = s_indices[0]
ts = t[ts_index]

# find the maximum of the horizontal envelope
peaks, _ = find_peaks(env_horizontal)
max_index = peaks[np.argmax(env_horizontal[peaks])]
t_max = t[max_index]

print("P-wave arrival:", tp, "s")
print("S-wave arrival:", ts, "s")
print("Maximum horizontal motion:", t_max, "s")

# plot arrival times
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

axes[0].plot(t, Z * 1000)
axes[0].set_ylabel("Z (mm/s)")

axes[1].plot(t, N * 1000)
axes[1].set_ylabel("N (mm/s)")

axes[2].plot(t, E * 1000)
axes[2].set_ylabel("E (mm/s)")
axes[2].set_xlabel("Time (s)")

for ax in axes:
    ax.axvline(tp, color="green", linestyle="--", label="P arrival")
    ax.axvline(ts, color="red", linestyle="--", label="S arrival")
    ax.legend()

plt.tight_layout()
plt.savefig("PW5/arrival_times.png")
plt.show()

# spectrogram of the raw Z channel
f, ts, S = signal.spectrogram(Z, fs=fs, nperseg=256)

plt.figure(figsize=(10, 5))
plt.pcolormesh(ts, f, 10 * np.log10(S), shading="auto")
plt.ylim(0, 10)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram of Z channel")
plt.colorbar(label="Power (dB)")

plt.tight_layout()
plt.savefig("PW5/spectrogram.png")
plt.show()

# magnitude calculation
PGV = np.max(np.sqrt(N**2 + E**2)) * 100

R = 306.3

M = (np.log10(PGV) + np.log10(R) + 1.0) / 0.55

print("PGV:", PGV, "cm/s")
print("Magnitude:", M)