import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# === USER SETTINGS ===
folder_path = "/Users/natalie/Nuclein/Engineering - General/Systems Integration/Heater + deltaF Investigation/05Aug lowdF Investigation"
sheet_name = 0
col_heater1 = 5  # 6th column (0-based indexing) → C3
col_heater2 = 6  # 7th column (0-based indexing) → C4
time_step = 0.1  # seconds between points
start_time_for_peak = 120  # seconds (for C4 peak detection)
c3_offset = 85  # seconds before C4 peak for C3 value
output_csv = os.path.join(folder_path, "heater_results.csv")
plots_folder = os.path.join(folder_path, "plots")

# Ensure plots folder exists
os.makedirs(plots_folder, exist_ok=True)

# === COLLECT RESULTS ===
results = []
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

for file_path in excel_files:
    print(f"Processing: {os.path.basename(file_path)}")
    
    # Skip first 6 rows of non-numeric data
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=6)

    # Extract data
    heater1 = pd.to_numeric(df.iloc[:, col_heater1], errors='coerce').dropna().values  # C3
    heater2 = pd.to_numeric(df.iloc[:, col_heater2], errors='coerce').dropna().values  # C4
    time = np.arange(len(heater1)) * time_step

    # Filter after start time for peak detection
    start_idx = int(start_time_for_peak / time_step)
    heater2_after = heater2[start_idx:]
    time_after = time[start_idx:]

    # Detect peak in C4
    peak_idx_h2 = np.argmax(heater2_after) + start_idx
    peak_val_h2 = heater2[peak_idx_h2]
    peak_time_h2 = time[peak_idx_h2]

    # Determine C3 value offset before C4 peak
    offset_idx_h1 = int(max(0, peak_idx_h2 - (c3_offset / time_step)))
    offset_val_h1 = heater1[offset_idx_h1]
    offset_time_h1 = time[offset_idx_h1]

    # Append results
    results.append([os.path.basename(file_path), offset_val_h1, offset_time_h1, peak_val_h2, peak_time_h2])

    # === PLOT AND SAVE ===
    plt.figure(figsize=(10, 6))
    plt.plot(time, heater1, label="C3 (Heater 1)", color="red")
    plt.plot(time, heater2, label="C4 (Heater 2)", color="blue")

    # Mark points
    plt.plot(offset_time_h1, offset_val_h1, "ro", label=f"C3 @ {offset_time_h1:.1f}s: {offset_val_h1:.2f}°C")
    plt.plot(peak_time_h2, peak_val_h2, "bo", label=f"C4 Peak @ {peak_time_h2:.1f}s: {peak_val_h2:.2f}°C")

    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Heater Data: {os.path.basename(file_path)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plots_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

# === SAVE RESULTS CSV ===
results_df = pd.DataFrame(results, columns=["Filename", "C3_Value (°C)", "C3_Time (s)",
                                            "C4_Peak (°C)", "C4_Peak_Time (s)"])
results_df.to_csv(output_csv, index=False)

print(f"\n✅ Processing complete. Results saved to: {output_csv}")
print(f"✅ Plots saved in: {plots_folder}")
