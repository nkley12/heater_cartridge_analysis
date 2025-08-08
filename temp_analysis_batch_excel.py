import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# === USER SETTINGS ===
input_folder = "/Users/natalie/Nuclein/Engineering - General/Systems Integration/Heater + deltaF Investigation/All heater data files"
threshold_temp = 68.5
max_slope = 7
xlim_start = 0
xlim_end = 120
analysis_start_time = 580  # <-- Ignore data before this time (seconds)

# === Prepare overall results container ===
summary_all_files = []

# === Process each XLSX file in the folder ===
for filename in os.listdir(input_folder):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(input_folder, filename)

        # Read Excel and extract column 5 (index 4), ignoring non-numeric rows
        df = pd.read_excel(file_path, header=None)
        temp_raw = pd.to_numeric(df.iloc[:, 4], errors='coerce').dropna().reset_index(drop=True)

        # Generate time vector starting at 0, incrementing by 0.1 seconds
        time = pd.Series(np.arange(0, len(temp_raw) * 0.1, 0.1))

        # === Filter data to start at analysis_start_time ===
        if any(time >= analysis_start_time):
            start_idx = time[time >= analysis_start_time].index[0]
        else:
            print(f"Skipping {filename}: no data beyond {analysis_start_time}s")
            continue  # Skip file if it doesn't reach the analysis start time

        time = time.iloc[start_idx:].reset_index(drop=True)
        temp_raw = temp_raw.iloc[start_idx:].reset_index(drop=True)

        # === Reset time to zero after trimming ===
        time = time - time.iloc[0]

        # Filter out spikes
        dt = np.gradient(time)
        dT = np.gradient(temp_raw)
        slope = np.abs(dT / dt)
        valid_mask = slope <= max_slope

        time_clean = time[valid_mask].reset_index(drop=True)
        temp_clean = temp_raw[valid_mask].reset_index(drop=True)

        # Detect peaks
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
                "Denature Temp": peak_temp,
                "Time of Peak": peak_time,
                "Anneal Start Temp": temp_clean.iloc[fall_below_idx] if fall_below_idx is not None else None,
                "Time Temp Falls Below 68": time_clean.iloc[fall_below_idx] if fall_below_idx is not None else None,
                "Image End Temp": min_temp,
                "Time of Min": min_time,
                "Anneal End Temp": temp_clean.iloc[drop2_idx] if drop2_idx is not None else None,
                "Time Pre-Imaging": time_clean.iloc[drop2_idx] if drop2_idx is not None else None,
                "Drop Time to Anneal": t_fall - peak_time if fall_below_idx is not None else np.nan,
                "Anneal Time": t_drop2 - t_fall if fall_below_idx is not None and drop2_idx is not None else np.nan,
                "Image Time": min_time - t_drop2 if drop2_idx is not None else np.nan
            })

        # DataFrame for this file
        results_df = pd.DataFrame(results)

        # Exclude first and last cycles for summary
        results_for_summary = results_df.iloc[1:-1].copy()

        # Calculate average cycle time between peaks
        if len(peaks) > 1:
            cycle_durations = np.diff([time_clean.iloc[p] for p in peaks])
            avg_cycle_time = np.mean(cycle_durations)
        else:
            avg_cycle_time = np.nan

        # Compute mean values
        mean_values = results_for_summary[[
            "Denature Temp",
            "Drop Time to Anneal",
            "Anneal Start Temp",
            "Anneal End Temp",
            "Anneal Time",
            "Image End Temp",
            "Image Time"
        ]].mean()

        # Store result in summary list
        summary_all_files.append({
            "Filename": filename,
            **mean_values.to_dict(),
             "Cycle Time": avg_cycle_time
        })

# === Save combined summary Excel file ===
summary_df = pd.DataFrame(summary_all_files)
output_excel = os.path.join(input_folder, "batch_summary_temps_and_times.xlsx")
summary_df.to_excel(output_excel, index=False, engine='openpyxl')
print(f"Saved batch summary Excel file to: {output_excel}")
