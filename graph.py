import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ---------------------------
# 1. Load the CSV and compute congestion measures per category
# ---------------------------
df = pd.read_csv("dataset.csv")
print(df.head())
print(df.columns.str.strip().str.lower())

# Define transportation type categories
transportation_types = {
    'Vehicle': ['vehicle', 'car', 'auto', 'passenger', 'truck', 'freight', 'commercial','bus'],
    'Bikes': ['bike', 'bicycle'],
    'Pedrestrains': ['pedrestrain', 'pedestrian', 'foot', 'walker','ped']
}

# Identify candidate columns and compute totals per category
transport_columns = {t: [] for t in transportation_types}
for col in df.columns:
    col_lower = col.lower()
    if ('volume' in col_lower or 'count' in col_lower or 
        any(keyword in col_lower for keywords in transportation_types.values() for keyword in keywords)):
        for transport_type, keywords in transportation_types.items():
            if any(keyword in col_lower for keyword in keywords):
                try:
                    pd.to_numeric(df[col])
                    transport_columns[transport_type].append(col)
                except Exception as e:
                    print(f"Skipping non-numeric column: {col}")

for transport_type, cols in transport_columns.items():
    if cols:
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[f'total_{transport_type.lower()}'] = df[cols].sum(axis=1)
        print(f"Using these columns for {transport_type}: {cols}")
    else:
        print(f"No suitable columns found for {transport_type}. Using random values for demonstration.")
        df[f'total_{transport_type.lower()}'] = np.random.randint(50, 500, size=len(df))

# ---------------------------
# 2. Process time-of-day from start_time and create 3‑hour bins
# ---------------------------
# Convert start_time (format: mm-dd-yyyyThh:mm:ss) to datetime
df['time'] = pd.to_datetime(df['start_time'], errors='coerce')

# Filter data to include only times between 07:30 and 18:00
start_time_filter = pd.to_datetime("07:30").time()
end_time_filter   = pd.to_datetime("18:00").time()
df = df[(df['time'].dt.time >= start_time_filter) & (df['time'].dt.time <= end_time_filter)]

# Function to bin times into 3‑hour intervals starting at 07:30.
def get_time_bin(dt):
    minutes = dt.hour * 60 + dt.minute
    if minutes < 450 or minutes > 1080:
        return None
    # Each bin is 180 minutes (3 hours); bins start at 07:30 (450 minutes)
    bin_index = (minutes - 450) // 180  
    bin_start = 450 + bin_index * 180  # will be 450, 630, 810, or 990 minutes
    hour = bin_start // 60
    minute = bin_start % 60
    return f"{hour:02d}:{minute:02d}"

df['time_bin'] = df['time'].apply(get_time_bin)
df = df.dropna(subset=['time_bin'])
unique_bins = sorted(df['time_bin'].unique())
print("Unique time bins:", unique_bins)

# Assign a distinct base color for each time bin (using a qualitative colormap)
base_cmap = plt.cm.get_cmap('tab10', len(unique_bins))
bin_to_color = {t_bin: base_cmap(i) for i, t_bin in enumerate(unique_bins)}

# ---------------------------
# 3. Compute a shade for each record per category
# ---------------------------
# For each time bin and for each category, we blend white and the base color according
# to the record's congestion value normalized within that time bin.
# We'll create three new columns: 'color_vehicle', 'color_bikes', 'color_pedrestrains'

def compute_shaded_color(series, base_color):
    """Compute an array of shaded colors by blending white and base_color based on normalized values."""
    vmin = series.min()
    vmax = series.max()
    # Avoid division by zero: if all values are equal, use norm=1.
    def blend(val):
        norm = 1 if vmax == vmin else (val - vmin) / (vmax - vmin)
        # norm=0 -> white; norm=1 -> base_color
        return norm * np.array(mcolors.to_rgb(base_color)) + (1 - norm) * np.array([1, 1, 1])
    return series.apply(lambda x: blend(x))

# For vehicles
df['color_vehicle'] = None
for t_bin in unique_bins:
    mask = df['time_bin'] == t_bin
    base_color = bin_to_color[t_bin]
    df.loc[mask, 'color_vehicle'] = compute_shaded_color(df.loc[mask, 'total_vehicle'], base_color)

# For bikes
df['color_bikes'] = None
for t_bin in unique_bins:
    mask = df['time_bin'] == t_bin
    base_color = bin_to_color[t_bin]
    df.loc[mask, 'color_bikes'] = compute_shaded_color(df.loc[mask, 'total_bikes'], base_color)

# For pedestrians
df['color_pedrestrains'] = None
for t_bin in unique_bins:
    mask = df['time_bin'] == t_bin
    base_color = bin_to_color[t_bin]
    df.loc[mask, 'color_pedrestrains'] = compute_shaded_color(df.loc[mask, 'total_pedrestrains'], base_color)

# ---------------------------
# 4. Plot three heatmaps (one per category)
# ---------------------------
def scale_size(val, max_val):
    return (val / max_val) * 100 + 20

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Congestion by Time Interval (3-hour bins) with Shades", fontsize=16)

# Vehicle heatmap
max_vehicle = df['total_vehicle'].max() if not df['total_vehicle'].empty else 1
axes[0].scatter(df["longitude"], df["latitude"],
                color=df['color_vehicle'].tolist(),
                s=df['total_vehicle'].apply(lambda x: scale_size(x, max_vehicle)),
                alpha=0.7)
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Vehicle Congestion")
axes[0].grid(True)

# Bikes heatmap
max_bikes = df['total_bikes'].max() if not df['total_bikes'].empty else 1
axes[1].scatter(df["longitude"], df["latitude"],
                color=df['color_bikes'].tolist(),
                s=df['total_bikes'].apply(lambda x: scale_size(x, max_bikes)),
                alpha=0.7)
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title("Bike Congestion")
axes[1].grid(True)

# Pedestrians heatmap
max_peds = df['total_pedrestrains'].max() if not df['total_pedrestrains'].empty else 1
axes[2].scatter(df["longitude"], df["latitude"],
                color=df['color_pedrestrains'].tolist(),
                s=df['total_pedrestrains'].apply(lambda x: scale_size(x, max_peds)),
                alpha=0.7)
axes[2].set_xlabel("Longitude")
axes[2].set_ylabel("Latitude")
axes[2].set_title("Pedestrian Congestion")
axes[2].grid(True)

# Create a common legend with one patch per time interval using the base colors
legend_patches = []
for t_bin in unique_bins:
    patch = mpatches.Patch(color=bin_to_color[t_bin], label=f"Time: {t_bin}")
    legend_patches.append(patch)
fig.legend(handles=legend_patches, title="Time Intervals", loc='upper center',
           ncol=len(unique_bins), bbox_to_anchor=(0.5, 0.95))

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()
print("run")
