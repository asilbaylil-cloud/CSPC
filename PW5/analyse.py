
import pandas as pd
import numpy as np
from scipy import signal


def analyse(csv_file):
    d = pd.read_csv(csv_file)

    t = d["time_s"].to_numpy()
    Z = d["Z"].to_numpy()
    N = d["N"].to_numpy()
    E = d["E"].to_numpy()

    fs = 1 / (t[1] - t[0])

    def bandpass(x, a, b):
        X = np.fft.fft(x)
        f = np.fft.fftfreq(len(x), d=1/fs)

        X[(np.abs(f) < a) | (np.abs(f) > b)] = 0

        return np.fft.ifft(X).real

    Z_clean = bandpass(Z, 0.5, 10)
    N_clean = bandpass(N, 0.5, 10)
    E_clean = bandpass(E, 0.5, 10)

    def envelope(x, window_seconds):
        w = int(window_seconds * fs)
        return np.convolve(np.abs(x), np.ones(w) / w, mode="same")

    env_Z = envelope(Z_clean, 0.05)
    env_horizontal = envelope(
        np.sqrt(N_clean**2 + E_clean**2), 0.5
    )

    p_limit = 0.05 * np.max(env_Z)
    p_indices = np.where(env_Z > p_limit)[0]
    tp = t[p_indices[0]]

    s_limit = 0.10 * np.max(env_horizontal)
    s_indices = np.where(
        (t > tp + 2) & (env_horizontal > s_limit)
    )[0]
    ts = t[s_indices[0]]

    peaks, _ = signal.find_peaks(env_horizontal)
    max_index = peaks[np.argmax(env_horizontal[peaks])]
    t_max = t[max_index]

    vp = 6.2
    vs = 3.6

    R = (ts - tp) / (1/vs - 1/vp)
    origin_time = tp - R / vp

    PGV = np.max(np.sqrt(N**2 + E**2)) * 100

    M = (np.log10(PGV) + np.log10(R) + 1.0) / 0.55

    return {
        "P_arrival": tp,
        "S_arrival": ts,
        "Maximum_motion": t_max,
        "Distance_km": R,
        "Origin_time": origin_time,
        "PGV_cm_s": PGV,
        "Magnitude": M
    }
import os

os.makedirs("PW5/results", exist_ok=True)

result_A = analyse("PW5/data/seismogram_A.csv")
result_B = analyse("PW5/data/seismogram_B.csv")

results = pd.DataFrame([result_A, result_B])
results.insert(0, "Earthquake", ["A", "B"])

results.to_csv("PW5/results/events.csv", index=False)

print(results)