import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# === USER SETTINGS ===
file_path = "/Users/natalie/OneDrive - Nuclein/Desktop/Heater data small test/C06250001_lowdf.xlsx"
threshold_temp = 68.5        # temperature drop threshold
xlim_start = 0               # x-axis min (s)
xlim_end = 120               # x-axis max (s)
max_slope = 7                # max °C/sec
analysis_start_time = 580     # ignore data before this time (seconds)

# === Load data ===
df = pd.read_excel(file_path, header=None)
temp_raw = pd.to_numeric(df.iloc[:, 4], errors='coerce').dropna().reset_index(drop=True)

# Generate time vector (0.1 s increments)
time = pd.Series(np.arange(0, len(temp_raw) * 0.1, 0.1))

# === Trim data to start at analysis_start_time ===
if any(time >= analysis_start_time):
    start_idx = time[time >= analysis_start_time].index[0]
else:
    raise ValueError(f"No data beyond {analysis_start_time}s in {file_path}")

time = time.iloc[start_idx:].reset_index(drop=True)
temp_raw = temp_raw.iloc[start_idx:].reset_index(drop=True)

# Reset time to zero after trimming
time = time - time.iloc[0]

# === Filter out spikes ===
dt = np.gradient(time)
dT = np.gradient(temp_raw)
slope = np.abs(dT / dt)
valid_mask = slope <= max_slope

time_clean = time[valid_mask].reset_index(drop=True)
temp_clean = temp_raw[valid_mask].reset_index(drop=True)

# === Detect peaks in cleaned data ===
peaks, _ = find_peaks(temp_clean, distance=50, prominence=1)

results = []
fall_below_idxs = []
min_idxs = []
drop2_idxs = []

for i, peak_idx in enumerate(peaks):
    peak_time = time_clean.iloc[peak_idx]
    peak_temp = temp_clean.iloc[peak_idx]

    # Find fall below threshold
    fall_below_idx = None
    for j in range(peak_idx + 1, len(temp_clean)):
        if temp_clean.iloc[j] < threshold_temp:
            fall_below_idx = j
            break
    fall_below_idxs.append(fall_below_idx)

    # Find min temp before next peak
    next_peak_idx = peaks[i + 1] if i + 1 < len(peaks) else len(temp_clean) - 1
    min_idx = temp_clean.iloc[peak_idx:next_peak_idx].idxmin()
    min_time = time_clean.iloc[min_idx]
    min_temp = temp_clean.iloc[min_idx]
    min_idxs.append(min_idx)

    # Detect pre-second-drop using derivative
    segment_temp = temp_clean.iloc[peak_idx:next_peak_idx]
    segment_time = time_clean.iloc[peak_idx:next_peak_idx]
    deriv = np.gradient(segment_temp, segment_time)

    drop2_idx = None
    for j in range(10, len(deriv) - 2):
        pre = np.mean(deriv[j - 2:j])
        post = np.mean(deriv[j + 1:j + 3])
        if pre > -0.2 and post < -0.5:
            drop2_idx = peak_idx + j
            break
    drop2_idxs.append(drop2_idx)

    # Extract time points for deltas
    t_fall = time_clean.iloc[fall_below_idx] if fall_below_idx is not None else np.nan
    t_drop2 = time_clean.iloc[drop2_idx] if drop2_idx is not None else np.nan

    results.append({
        "Cycle #": i + 1,
        "Denature Peak Temp": peak_temp,
        "Time of Peak": peak_time,
        "Temp Falls Below 68": temp_clean.iloc[fall_below_idx] if fall_below_idx is not None else None,
        "Time Temp Falls Below 68": time_clean.iloc[fall_below_idx] if fall_below_idx is not None else None,
        "Min Imaging Temp": min_temp,
        "Time of Min": min_time,
        "Pre-Imaging Temp": temp_clean.iloc[drop2_idx] if drop2_idx is not None else None,
        "Time Pre-Imaging": time_clean.iloc[drop2_idx] if drop2_idx is not None else None,
        "Time_Peak_to_68": t_fall - peak_time if fall_below_idx is not None else np.nan,
        "Time_68_to_PreImaging": t_drop2 - t_fall if fall_below_idx is not None and drop2_idx is not None else np.nan,
        "Time_PreImaging_to_Min": min_time - t_drop2 if drop2_idx is not None else np.nan
    })

# === Save results with summary ===
results_df = pd.DataFrame(results)

# Exclude first and last cycles for summary statistics
results_for_summary = results_df.iloc[1:-1].copy()

# Compute summary stats on middle cycles
summary_stats = results_for_summary[[
    "Denature Peak Temp",
    "Temp Falls Below 68",
    "Pre-Imaging Temp",
    "Min Imaging Temp",
    "Time_Peak_to_68",
    "Time_68_to_PreImaging",
    "Time_PreImaging_to_Min"
]].agg(['mean', 'std'])

# Add summary rows to DataFrame
summary_rows = pd.DataFrame({
    "Cycle #": ["Mean", "Standard Deviation"],
    **{col: [summary_stats.loc['mean', col], summary_stats.loc['std', col]] for col in summary_stats.columns}
})

# Append summary to the result
results_with_summary = pd.concat([results_df, summary_rows], ignore_index=True)

# Save to CSV
output_folder = os.path.dirname(file_path)
input_filename = os.path.splitext(os.path.basename(file_path))[0]
output_file = os.path.join(output_folder, f"{input_filename}_temps_and_times.csv")
results_with_summary.to_csv(output_file, index=False)

print(f"Saved full results with summary to: {output_file}")

# === Plot ===
plt.figure(figsize=(12, 6))

# Plot raw data
plt.plot(time, temp_raw, label="Raw Temperature", color="lightgray", linewidth=1)

# Plot filtered (cleaned) data
plt.plot(time_clean, temp_clean, label="Filtered Temperature", color="black", linewidth=2)

# Mark detected points on filtered data
plt.plot(time_clean.iloc[peaks], temp_clean.iloc[peaks], 'ro', label="Peak")
plt.plot([time_clean.iloc[i] for i in fall_below_idxs if i is not None], 
         [temp_clean.iloc[i] for i in fall_below_idxs if i is not None], 'bo', label="Fall Below Threshold")
plt.plot([time_clean.iloc[i] for i in min_idxs if i is not None], 
         [temp_clean.iloc[i] for i in min_idxs if i is not None], 'go', label="Min Temp")
plt.plot([time_clean.iloc[i] for i in drop2_idxs if i is not None], 
         [temp_clean.iloc[i] for i in drop2_idxs if i is not None], 'mo', label="Pre-Second Drop")

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Raw vs Filtered Temperature w/Temps Labeled")
plt.xlim(xlim_start, xlim_end)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
