import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file from dataset.csv instead of the tmc_ file
df = pd.read_csv("dataset.csv")

print(df.head())
print(df.columns.str.strip().str.lower())

# Define the transportation type categories in the desired order:
# Vehicle, Bikes, Pedrestrains.
# Adjust the keywords according to the column names in your dataset.
transportation_types = {
    'Vehicle': ['vehicle', 'car', 'auto', 'passenger', 'truck', 'freight', 'commercial'],
    'Bikes': ['bike', 'bicycle'],
    'Pedrestrains': ['pedrestrain', 'pedestrian', 'foot', 'walker', 'peds']
}

# Initialize a dictionary to store matching columns for each category
transport_columns = {transport_type: [] for transport_type in transportation_types}

# Find numeric columns that might relate to each transportation type.
# Here, we check if the column name contains common volume/count indicators and one of the keywords.
for col in df.columns:
    col_lower = col.lower()
    if 'volume' in col_lower or 'count' in col_lower or any(keyword in col_lower for keywords in transportation_types.values() for keyword in keywords):
        for transport_type, keywords in transportation_types.items():
            if any(keyword in col_lower for keyword in keywords):
                try:
                    # Verify it's numeric
                    pd.to_numeric(df[col])
                    transport_columns[transport_type].append(col)
                except:
                    print(f"Skipping non-numeric column: {col}")

# Calculate total volume (or count) for each transportation type by summing the relevant columns.
for transport_type, columns in transport_columns.items():
    if columns:
        print(f"Using these columns for {transport_type}: {columns}")
        # Convert columns to numeric
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Sum the columns for this transportation type
        df[f'total_{transport_type.lower()}'] = df[columns].sum(axis=1)
    else:
        # If no columns match, create random data for demonstration purposes.
        print(f"No suitable columns found for {transport_type}. Using random values for demonstration.")
        df[f'total_{transport_type.lower()}'] = np.random.randint(50, 500, size=len(df))

# Create a figure with 3 subplots for the 3 categories in the specified order.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Traffic/Usage Visualization by Category", fontsize=16)

# Plot each transportation type as a scatter plot, using longitude and latitude for position.
# The point size and color intensity reflect the computed totals.
for transport_type, ax in zip(transportation_types.keys(), axes):
    column_name = f'total_{transport_type.lower()}'
    
    # Create scatter plot
    scatter = ax.scatter(
        df["longitude"], 
        df["latitude"],
        c=df[column_name],  # Color by the computed total
        s=(df[column_name] / max(df[column_name].max(), 1)) * 100 + 20,  # Scale point size by the total
        cmap='YlOrRd',  # Colormap choice
        alpha=0.7
    )
    
    # Add a colorbar to each subplot
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'{transport_type} Count')
    
    # Set plot labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{transport_type} Data")
    ax.grid(True)
    
    # Annotate the top 3 points (largest totals) on each plot
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
plt.subplots_adjust(top=0.85)  # Adjust for the super title
plt.show()
print("run")
