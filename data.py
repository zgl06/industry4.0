import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv("tmc_most_recent_summary_data.csv")

print(df.head())
print(df.columns.str.strip().str.lower())

# Look for potential volume columns
volume_cols = [col for col in df.columns if 'volume' in col.lower() or 'count' in col.lower()]

# Check which columns are numeric
numeric_cols = []
for col in volume_cols:
    try:
        # Try to convert to numeric, will succeed only if column contains numbers or NaN
        pd.to_numeric(df[col])
        numeric_cols.append(col)
    except:
        print(f"Skipping non-numeric column: {col}")

# If we found numeric volume columns, use them
if numeric_cols:
    print(f"Using these columns for congestion: {numeric_cols}")
    # Convert all selected columns to numeric to be safe (will convert strings to NaN)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sum only the numeric columns
    df['total_vehicles'] = df[numeric_cols].sum(axis=1)
else:
    # If no suitable columns were found, create a random measure for demonstration
    print("No suitable numeric traffic volume columns found. Using random values for demonstration.")
    df['total_vehicles'] = np.random.randint(100, 1000, size=len(df))

# Create the plot
plt.figure(figsize=(12, 8))

# Plot points with color based on congestion
scatter = plt.scatter(
    df["longitude"], 
    df["latitude"],
    c=df['total_vehicles'],  # Color by congestion
    s=df['total_vehicles'] / max(df['total_vehicles'].max(), 1) * 100 + 20,  # Size by congestion
    cmap='YlOrRd',  # Yellow-Orange-Red colormap
    alpha=0.7
)

# Add colorbar for reference
cbar = plt.colorbar(scatter)
cbar.set_label('Congestion Level (Total Vehicles)')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Traffic Congestion Map")
plt.grid(True)

# Add some annotations for the most congested areas
top_congested = df.nlargest(5, 'total_vehicles')
for idx, row in top_congested.iterrows():
    plt.annotate(
        f"High: {int(row['total_vehicles'])}",
        (row['longitude'], row['latitude']),
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

plt.tight_layout()
plt.show()