import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import branca.colormap as cm
import seaborn as sns
from datetime import datetime, timedelta

# Function to process the data
def process_data(data):
    # Ensure data is in the right format
    if isinstance(data, str):
        try:
            data = pd.read_csv(data)
        except:
            print("Please provide a valid DataFrame or path to CSV file")
            return None
    
    # Convert date-time columns if they exist
    date_cols = ['count_date', 'start_time', 'end_time']
    for col in date_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Calculate total vehicles for each entry
    vehicle_cols = [col for col in data.columns if any(x in col for x in ['cars', 'truck', 'bus'])]
    data['total_vehicles'] = data[vehicle_cols].sum(axis=1)
    
    # Calculate congestion index
    # For simplicity, we'll use total vehicles as the main factor
    # but you can make this more sophisticated
    data['congestion_index'] = data['total_vehicles']
    
    # Normalize to 0-100 scale
    max_congestion = data['congestion_index'].max()
    data['congestion_index'] = (data['congestion_index'] / max_congestion) * 100
    
    return data

# Function to create an interactive map
def create_traffic_map(data, title="Traffic Congestion Map"):
    # Check if longitude and latitude are present
    if 'longitude' not in data.columns or 'latitude' not in data.columns:
        print("Data must contain 'longitude' and 'latitude' columns")
        return None
    
    # Create a map centered on the mean coordinates
    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    traffic_map = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                            tiles='cartodbpositron')
    
    # Create a colormap for congestion levels
    colormap = cm.linear.YlOrRd_09.scale(0, 100)
    colormap.caption = 'Congestion Index (0-100)'
    traffic_map.add_child(colormap)
    
    # Group by location to get average congestion
    if 'location_name' in data.columns:
        location_data = data.groupby(['location_name', 'latitude', 'longitude'])['congestion_index'].mean().reset_index()
    else:
        location_data = data.groupby(['latitude', 'longitude'])['congestion_index'].mean().reset_index()
    
    # Add markers for each location
    for idx, row in location_data.iterrows():
        # Determine circle color based on congestion level
        color = colormap(row['congestion_index'])
        
        # Create popup content
        if 'location_name' in row:
            popup_content = f"""
            <b>Location:</b> {row['location_name']}<br>
            <b>Congestion Index:</b> {row['congestion_index']:.1f}<br>
            <b>Coordinates:</b> {row['latitude']:.5f}, {row['longitude']:.5f}
            """
        else:
            popup_content = f"""
            <b>Congestion Index:</b> {row['congestion_index']:.1f}<br>
            <b>Coordinates:</b> {row['latitude']:.5f}, {row['longitude']:.5f}
            """
        
        # Add circle marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10 + (row['congestion_index'] / 10),  # Size based on congestion
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(traffic_map)
    
    # Add heat map layer
    heat_data = [[row['latitude'], row['longitude'], row['congestion_index']] 
                 for idx, row in location_data.iterrows()]
    
    HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'yellow', 1: 'red'}).add_to(traffic_map)
    
    # Add title
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    traffic_map.get_root().html.add_child(folium.Element(title_html))
    
    return traffic_map

# Function to create time-based maps
def create_time_based_maps(data, time_intervals=None):
    """
    Create separate maps for different time periods
    
    Parameters:
    - data: DataFrame with traffic data
    - time_intervals: List of hour ranges to create maps for
                     e.g. [(7,9), (12,14), (17,19)]
    
    Returns:
    - Dictionary of maps for each time period
    """
    if 'start_time' not in data.columns:
        print("Data must contain 'start_time' column")
        return None
    
    if time_intervals is None:
        # Default time intervals for morning rush, midday, evening rush
        time_intervals = [(7,9), (11,14), (16,19)]
    
    maps = {}
    
    for start_hour, end_hour in time_intervals:
        # Filter data for the time period
        period_data = data[data['start_time'].dt.hour.between(start_hour, end_hour)]
        
        if len(period_data) > 0:
            period_name = f"{start_hour:02d}:00-{end_hour:02d}:00"
            maps[period_name] = create_traffic_map(
                period_data, 
                title=f"Traffic Congestion ({period_name})"
            )
    
    return maps

# Function to visualize congestion patterns over time for a specific location
def plot_location_time_patterns(data, location_name=None):
    """
    Create a time series plot of congestion for a specific location
    or average across all locations
    """
    if 'start_time' not in data.columns:
        print("Data must contain 'start_time' column")
        return
    
    # Make a copy to avoid modifying the original
    plot_data = data.copy()
    
    # Extract hour from start_time
    plot_data['hour'] = plot_data['start_time'].dt.hour
    
    plt.figure(figsize=(14, 8))
    
    if location_name and 'location_name' in plot_data.columns:
        # Filter for the specific location
        location_data = plot_data[plot_data['location_name'] == location_name]
        
        if len(location_data) == 0:
            print(f"No data found for location: {location_name}")
            return
        
        # Group by hour and get average congestion
        hourly_data = location_data.groupby('hour')['congestion_index'].mean()
        
        sns.lineplot(x=hourly_data.index, y=hourly_data.values, marker='o', linewidth=2)
        plt.title(f"Average Congestion by Hour of Day - {location_name}")
    else:
        # Group by hour and get average congestion across all locations
        hourly_data = plot_data.groupby('hour')['congestion_index'].mean()
        
        sns.lineplot(x=hourly_data.index, y=hourly_data.values, marker='o', linewidth=2)
        plt.title("Average Congestion by Hour of Day - All Locations")
    
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Congestion Index")
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt

# Function to create a 3D visualization of the data
def create_3d_time_location_plot(data):
    """
    Create a 3D plot with x=longitude, y=latitude, z=time, and color=congestion
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract hour from start_time
    data['hour'] = data['start_time'].dt.hour
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(
        data['longitude'], 
        data['latitude'], 
        data['hour'],
        c=data['congestion_index'], 
        cmap='YlOrRd',
        s=50 + data['congestion_index']/2,  # Size based on congestion
        alpha=0.7
    )
    
    # Add labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Hour of Day')
    ax.set_title('3D Visualization of Traffic Congestion (Location vs Time)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Congestion Index')
    
    plt.tight_layout()
    return plt

# Example usage
# Load your data
data = pd.read_csv("University_Dataset(tmc_raw_data_2020_2029).csv")

# Process the data
processed_data = process_data(data)

# Create an interactive map
traffic_map = create_traffic_map(processed_data)
traffic_map.save("traffic_congestion_map.html")  # Saves as an interactive HTML file

# Create time-based maps for different periods
time_maps = create_time_based_maps(processed_data)
for period, map_obj in time_maps.items():
    map_obj.save(f"traffic_map_{period}.html")

# Plot time patterns for a specific intersection
plot_location_time_patterns(processed_data, "Erindale Ave / Broadview Ave")