import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === USER SETTINGS ===
file_path = "//nuc-fs1/Engineering/Grant/DASH/Cartridge/Heater Test Cartridge/2025_07_08 dF Investigation/C08230002_1.xlsx"
sheet_name = 0  # Change if needed
col_heater1 = 5  # 6th column (0-based indexing)
col_heater2 = 6  # 7th column (0-based indexing)
time_step = 0.1  # seconds between points
start_time_for_peak = 150  # seconds

# === LOAD DATA ===
df = pd.read_excel(file_path, sheet_name=sheet_name)
heater1 = df.iloc[:, col_heater1].dropna().values
heater2 = df.iloc[:, col_heater2].dropna().values
time = np.arange(len(heater1)) * time_step

# Filter data after the start time
start_idx = int(start_time_for_peak / time_step)
heater1_after = heater1[start_idx:]
heater2_after = heater2[start_idx:]
time_after = time[start_idx:]

# === DETECT PEAKS ===
peak_idx_h1 = np.argmax(heater1_after) + start_idx
peak_val_h1 = heater1[peak_idx_h1]

peak_idx_h2 = np.argmax(heater2_after) + start_idx
peak_val_h2 = heater2[peak_idx_h2]

# === REPORT RESULTS ===
print(f"C3 (peak): {peak_val_h1:.2f} °C at {time[peak_idx_h1]:.1f} s")
print(f"C4 (peak): {peak_val_h2:.2f} °C at {time[peak_idx_h2]:.1f} s")

# === PLOT ===
plt.figure(figsize=(10, 6))
plt.plot(time, heater1, label="C3", color="red")
plt.plot(time, heater2, label="C4", color="blue")

# Mark detected peaks
plt.plot(time[peak_idx_h1], peak_val_h1, "ro")
plt.plot(time[peak_idx_h2], peak_val_h2, "bo")

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Heater Temperature Data with Peaks Detected")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
