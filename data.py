import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.image as mpimg
import contextily as ctx
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Load the CSV file
df = pd.read_csv("tmc_most_recent_summary_data.csv")

print("Dataset overview:")
print(df.head())
print("\nColumns in dataset:")
print(df.columns.tolist())

# Look for date columns
date_columns = [col for col in df.columns if any(date_word in col.lower() for date_word in ['date', 'time', 'day'])]
print(f"\nPotential date columns found: {date_columns}")

# Process date column
if date_columns:
    date_col = date_columns[0]
    
    # Try to convert to datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Create a separate column for date only
        df['date_only'] = df[date_col].dt.date
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['date_only'])
        
        # Get unique dates for slider
        unique_dates = sorted(df['date_only'].unique())
        print(f"Found {len(unique_dates)} unique dates in the dataset")
        
        if len(unique_dates) == 0:
            raise ValueError("No valid dates found after conversion")
            
    except Exception as e:
        print(f"Error processing dates: {e}")
        print("Creating artificial date range for demonstration")
        # Create artificial dates
        base_date = pd.Timestamp('2023-01-01')
        df['artificial_date'] = [(base_date + timedelta(days=i % 30)) for i in range(len(df))]
        df['date_only'] = [d.date() for d in df['artificial_date']]
        date_col = 'artificial_date'
        unique_dates = sorted(df['date_only'].unique())
else:
    print("No date columns found. Creating artificial dates for demonstration.")
    base_date = pd.Timestamp('2023-01-01')
    df['artificial_date'] = [(base_date + timedelta(days=i % 30)) for i in range(len(df))]
    df['date_only'] = [d.date() for d in df['artificial_date']]
    date_col = 'artificial_date'
    unique_dates = sorted(df['date_only'].unique())

# Examine actual traffic volume columns
print("\nExamining potential traffic volume columns:")
potential_traffic_cols = ['total_vehicle', 'am_peak_vehicle', 'pm_peak_vehicle', 
                         'n_appr_vehicle', 'e_appr_vehicle', 's_appr_vehicle', 'w_appr_vehicle']

# Check which columns exist in the dataset
volume_cols = [col for col in potential_traffic_cols if col in df.columns]
print(f"Found these traffic volume columns: {volume_cols}")

# Find numeric volume columns
numeric_cols = []
for col in volume_cols:
    try:
        pd.to_numeric(df[col])
        numeric_cols.append(col)
        print(f"Verified {col} is numeric")
    except:
        print(f"Skipping non-numeric column: {col}")

# If we found numeric volume columns, use them
if numeric_cols:
    print(f"Using these columns for congestion: {numeric_cols}")
    # Convert all selected columns to numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sum the numeric columns
    df['total_vehicles'] = df[numeric_cols].sum(axis=1)
else:
    # If no suitable columns were found, create random values
    print("No suitable numeric traffic volume columns found. Using random values for demonstration.")
    df['total_vehicles'] = np.random.randint(100, 1000, size=len(df))

# Print the range of date points
print(f"\nDate range in the dataset: {min(unique_dates)} to {max(unique_dates)}")

# Count data points per date to identify sparse days
date_counts = df.groupby('date_only').size()
print(f"\nAverage data points per day: {date_counts.mean():.1f}")
print(f"Minimum data points on a day: {date_counts.min()} (on {date_counts.idxmin()})")
print(f"Maximum data points on a day: {date_counts.max()} (on {date_counts.idxmax()})")

# For sparse data, we want to aggregate nearby days
# Define a date window size (in days) to show more points if a day has sparse data
WINDOW_SIZE = 7  # Will show 3 days before and 3 days after if needed

# Define the Toronto bounding box (approximate)
# These values represent the approximate lat/long boundaries of Toronto
TORONTO_BOUNDS = {
    'min_lat': 43.58,
    'max_lat': 43.85,
    'min_lon': -79.62,
    'max_lon': -79.15
}

# Check if data points are within Toronto bounds
in_bounds = ((df['latitude'] >= TORONTO_BOUNDS['min_lat']) & 
             (df['latitude'] <= TORONTO_BOUNDS['max_lat']) & 
             (df['longitude'] >= TORONTO_BOUNDS['min_lon']) & 
             (df['longitude'] <= TORONTO_BOUNDS['max_lon']))

if not in_bounds.all():
    print(f"\nWarning: {(~in_bounds).sum()} data points are outside Toronto boundaries.")
    # Filter to keep only points inside Toronto
    df = df[in_bounds]
    print(f"Keeping {len(df)} data points within Toronto boundaries.")

# Create interactive figure with slider
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.25)  # Make room for slider

# Try to add a contextily basemap for Toronto
try:
    import contextily as ctx
    has_contextily = True
    print("Using contextily for map background")
except ImportError:
    has_contextily = False
    print("Contextily not available. Install with 'pip install contextily' for map background")

# Create initial empty scatter plot
scatter = ax.scatter([], [], s=[], c=[], cmap='YlOrRd', alpha=0.7)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Congestion Level (Total Vehicles)')

# Add title that will be updated
title = ax.set_title("Traffic Congestion Map")

# Set initial map boundaries to Toronto
ax.set_xlim(TORONTO_BOUNDS['min_lon'], TORONTO_BOUNDS['max_lon'])
ax.set_ylim(TORONTO_BOUNDS['min_lat'], TORONTO_BOUNDS['max_lat'])

# Text annotations for congested areas (will be updated)
annotations = []

# Add slider for date selection
ax_date = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
date_idx_slider = Slider(
    ax=ax_date,
    label='Date',
    valmin=0,
    valmax=len(unique_dates) - 1,
    valinit=0,
    valstep=1
)

# Format the slider labels with actual dates
date_formatter = lambda x: str(unique_dates[int(x)]) if x < len(unique_dates) else ""
date_idx_slider.valtext.set_text(date_formatter(0))

# Function to update the plot based on slider value
def update(val):
    # Clear previous annotations
    for ann in annotations:
        ann.remove()
    annotations.clear()
    
    # Get selected date index
    date_idx = int(date_idx_slider.val)
    selected_date = unique_dates[date_idx]
    
    # Update slider text to show the date
    date_idx_slider.valtext.set_text(date_formatter(date_idx))
    
    # Filter data for the selected date
    day_data = df[df['date_only'] == selected_date]
    
    # Check if we have sparse data for this day (fewer than 10 points)
    if len(day_data) < 10:
        print(f"Sparse data ({len(day_data)} points) for {selected_date}, expanding window...")
        
        # Find date indices within window
        selected_idx = unique_dates.index(selected_date)
        start_idx = max(0, selected_idx - WINDOW_SIZE//2)
        end_idx = min(len(unique_dates)-1, selected_idx + WINDOW_SIZE//2)
        
        # Get date range
        date_range = unique_dates[start_idx:end_idx+1]
        
        # Get data for the date range
        day_data = df[df['date_only'].isin(date_range)]
        
        # Update title to reflect the date range
        title_text = f"Traffic Congestion Map - {selected_date} (+ nearby dates, {len(day_data)} points)"
    else:
        title_text = f"Traffic Congestion Map - {selected_date} ({len(day_data)} points)"
    
    if len(day_data) == 0:
        print(f"Warning: No data found for date {selected_date} even with expanded window")
        scatter.set_offsets(np.zeros((0, 2)))
        scatter.set_sizes(np.array([]))
        scatter.set_array(np.array([]))
        title.set_text(f"Traffic Congestion Map - {selected_date} (No Data)")
        fig.canvas.draw_idle()
        return
    
    # Update scatter plot data
    scatter.set_offsets(np.c_[day_data['longitude'], day_data['latitude']])
    
    # Update sizes (normalize to avoid extremely large points)
    max_val = max(day_data['total_vehicles'].max(), 1)
    sizes = day_data['total_vehicles'] / max_val * 100 + 20
    scatter.set_sizes(sizes)
    
    # Update colors
    scatter.set_array(day_data['total_vehicles'])
    
    # Update colorbar limits
    scatter.set_clim(day_data['total_vehicles'].min(), day_data['total_vehicles'].max())
    
    # Update title
    title.set_text(title_text)
    
    # Add annotations for top congested areas
    top_congested = day_data.nlargest(5, 'total_vehicles')
    for idx, row in top_congested.iterrows():
        ann = ax.annotate(
            f"High: {int(row['total_vehicles'])}",
            (row['longitude'], row['latitude']),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        annotations.append(ann)
    
    # Keep Toronto boundaries
    ax.set_xlim(TORONTO_BOUNDS['min_lon'], TORONTO_BOUNDS['max_lon'])
    ax.set_ylim(TORONTO_BOUNDS['min_lat'], TORONTO_BOUNDS['max_lat'])
    
    # Redraw the figure
    fig.canvas.draw_idle()

# Try to add the Toronto map background
try:
    if has_contextily:
        # Add a Toronto basemap using contextily
        ctx.add_basemap(
            ax, 
            crs='EPSG:4326',  # WGS84 coordinate system (standard lat/long)
            source=ctx.providers.CartoDB.Positron
        )
        print("Added Toronto map background")
except Exception as e:
    print(f"Could not add map background: {e}")
    print("Continuing without map background...")

# Register the update function with the slider
date_idx_slider.on_changed(update)

# Initialize with first date
update(0)

# Function to handle keyboard navigation
def key_press(event):
    if event.key == 'right':
        new_val = min(date_idx_slider.val + 1, date_idx_slider.valmax)
        date_idx_slider.set_val(new_val)
    elif event.key == 'left':
        new_val = max(date_idx_slider.val - 1, date_idx_slider.valmin)
        date_idx_slider.set_val(new_val)

fig.canvas.mpl_connect('key_press_event', key_press)

# Add instructions text
plt.figtext(0.5, 0.01, "Use slider to change date or left/right arrow keys", 
            ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Adjust layout to make room for slider
plt.show()

# Save functionality
def save_current_view():
    current_date = unique_dates[int(date_idx_slider.val)]
    filename = f"congestion_map_{current_date}.png"
    fig.savefig(filename)
    print(f"Saved current view as {filename}")

# Uncomment to add save button
# from matplotlib.widgets import Button
# ax_save = plt.axes([0.8, 0.05, 0.1, 0.04])
# save_button = Button(ax_save, 'Save View')
# save_button.on_clicked(lambda event: save_current_view())