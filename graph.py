import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv("tmc_most_recent_summary_data.csv")

print(df.head())
print(df.columns.str.strip().str.lower())

# Determine transportation types from the dataset
# Since we don't know the exact column names, we'll look for common keywords
# Assuming we want to separate by: cars, trucks, and buses/public transport

# Define transportation type categories and their related keywords
transportation_types = {
    'Cars': ['car', 'passenger', 'auto', 'personal'],
    'Trucks': ['truck', 'freight', 'delivery', 'commercial'],
    'Public Transport': ['bus', 'transit', 'public', 'transport']
}

# Initialize dictionaries to store columns for each transportation type
transport_columns = {transport_type: [] for transport_type in transportation_types}

# Find volume columns that match each transportation type
for col in df.columns:
    col_lower = col.lower()
    if 'volume' in col_lower or 'count' in col_lower:
        # Check which transportation type this column belongs to
        for transport_type, keywords in transportation_types.items():
            if any(keyword in col_lower for keyword in keywords):
                try:
                    # Verify it's numeric
                    pd.to_numeric(df[col])
                    transport_columns[transport_type].append(col)
                except:
                    print(f"Skipping non-numeric column: {col}")

# Calculate total volume for each transportation type
for transport_type, columns in transport_columns.items():
    if columns:
        print(f"Using these columns for {transport_type}: {columns}")
        # Convert columns to numeric
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sum the columns for this transportation type
        df[f'total_{transport_type.lower().replace(" ", "_")}'] = df[columns].sum(axis=1)
    else:
        # If no suitable columns found, create random data for demonstration
        print(f"No suitable columns found for {transport_type}. Using random values for demonstration.")
        df[f'total_{transport_type.lower().replace(" ", "_")}'] = np.random.randint(50, 500, size=len(df))

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Traffic Congestion by Transportation Type", fontsize=16)

# Plot each transportation type
for i, (transport_type, ax) in enumerate(zip(transportation_types.keys(), axes)):
    column_name = f'total_{transport_type.lower().replace(" ", "_")}'
    
    # Create the scatter plot
    scatter = ax.scatter(
        df["longitude"], 
        df["latitude"],
        c=df[column_name],  # Color by congestion
        s=df[column_name] / max(df[column_name].max(), 1) * 100 + 20,  # Size by congestion
        cmap='YlOrRd',  # Yellow-Orange-Red colormap
        alpha=0.7
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'{transport_type} Congestion')
    
    # Set labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{transport_type} Congestion")
    ax.grid(True)
    
    # Add annotations for most congested areas
    top_congested = df.nlargest(3, column_name)
    for idx, row in top_congested.iterrows():
        ax.annotate(
            f"High: {int(row[column_name])}",
            (row['longitude'], row['latitude']),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )

plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Make room for the super title
plt.show()
print("run")