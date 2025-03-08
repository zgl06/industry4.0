import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import folium
import math

###############################################################################
# 1. Data Processing: read CSV, compute congestion index
###############################################################################
def process_data(csv_path):
    """
    Reads the CSV file, converts datetime columns, and calculates a normalized
    congestion index (0-100) for each row based on total vehicle counts.
    Expects columns:
        - 'latitude', 'longitude' for location
        - possibly 'start_time' (for time-based filtering)
        - vehicle count columns with strings like 'car', 'truck', or 'bus'
    """
    data = pd.read_csv(csv_path)

    # Convert potential datetime columns if they exist
    for col in ['count_date', 'start_time', 'end_time']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Identify vehicle count columns (case-insensitive for 'car', 'truck', 'bus')
    vehicle_cols = [col for col in data.columns if any(x in col.lower() for x in ['car', 'truck', 'bus'])]
    if not vehicle_cols:
        print("Warning: No vehicle count columns found!")
    data['total_vehicles'] = data[vehicle_cols].sum(axis=1)
    
    # Normalize congestion index to 0-100
    max_val = data['total_vehicles'].max() or 1
    data['congestion_index'] = (data['total_vehicles'] / max_val) * 100
    return data

###############################################################################
# 2. Download Road Network with OSMnx (using bounding box)
###############################################################################
def load_osm_graph_bbox():
    """
    Loads a drivable road network within a bounding box around downtown Toronto.
    Adjust bounding box coordinates and custom_filter as needed.
    """
    # Define bounding box for your area of interest
    north = 43.70
    south = 43.66
    east = -79.36
    west = -79.40
    bbox = (north, south, east, west)
    
    # Optionally, specify road types to reduce data volume:
    custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]'
    
    print(f"Downloading OSM road network for bbox: {bbox} with custom filter: {custom_filter}")
    # For some OSMnx versions, graph_from_bbox expects bounding box as positional args
    # If your OSMnx is older/newer, adjust accordingly
    G = ox.graph_from_bbox(
        north, south, east, west,
        network_type='drive',
        custom_filter=custom_filter
    )
    return G

###############################################################################
# 3. Snap Traffic Data to Nearest Nodes & Aggregate Congestion
###############################################################################
def map_congestion_to_nodes(data, G):
    """
    For each row in 'data' (with lat/long and a 'congestion_index'),
    find the nearest node in G (using (lon, lat)) and store that in 'nearest_node'.
    Then compute the average congestion for each node.
    Returns a dict: node_id -> average congestion index.
    """
    node_ids = []
    for _, row in data.iterrows():
        nearest_node = ox.distance.nearest_nodes(G, row['longitude'], row['latitude'])
        node_ids.append(nearest_node)
    data['nearest_node'] = node_ids
    
    # Group by node and compute average congestion
    node_congestion = data.groupby('nearest_node')['congestion_index'].mean().to_dict()
    return node_congestion

def update_graph_congestion(G, node_congestion):
    """
    Adds a 'congestion' attribute to each node in G,
    defaulting to 0 if that node is not in node_congestion.
    """
    for node in G.nodes:
        G.nodes[node]['congestion'] = node_congestion.get(node, 0)
    return G

###############################################################################
# 4. Update Edge Weights
###############################################################################
def update_edge_weights(G, alpha=0.5):
    """
    For each edge, combine physical distance (in km) with the average congestion
    of its endpoint nodes. The 'weight' is used by Dijkstra to find the best route.
    
    weight = distance_km * (1 + alpha * (mean_congestion / 100))
    """
    for u, v, d in G.edges(data=True):
        length_m = d.get('length', 1)  # length in meters
        dist_km = length_m / 1000.0
        
        cong_u = G.nodes[u].get('congestion', 0)
        cong_v = G.nodes[v].get('congestion', 0)
        mean_cong = (cong_u + cong_v) / 2
        
        d['weight'] = dist_km * (1 + alpha * (mean_cong / 100))
    return G

###############################################################################
# 5. Shortest Path Computation (Neighbor-Only)
###############################################################################
def compute_shortest_path(G, orig_node, dest_node):
    """
    Uses Dijkstra's algorithm (networkx.shortest_path) with our custom 'weight' attribute.
    The path is guaranteed to only traverse legitimate edges between neighbor nodes.
    """
    try:
        path = nx.shortest_path(G, source=orig_node, target=dest_node, weight='weight')
        return path
    except nx.NetworkXNoPath:
        print("No available path between the selected nodes.")
        return None

###############################################################################
# 6. Visualization: use edge geometry so the route follows roads
###############################################################################
def create_route_map(G, route, title="Optimal Route"):
    """
    Draws the route on a Folium map. If an edge has a 'geometry' (LineString),
    we plot its actual shape. Otherwise, we fall back to a straight line
    between node coordinates.
    """
    if not route or len(route) < 2:
        print("Route is empty or invalid. Cannot visualize.")
        return None
    
    full_route_coords = []
    
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]
        
        # Default: straight line from node to node
        u_lat = G.nodes[u].get('y', 0)
        u_lon = G.nodes[u].get('x', 0)
        v_lat = G.nodes[v].get('y', 0)
        v_lon = G.nodes[v].get('x', 0)
        segment_coords = [(u_lat, u_lon), (v_lat, v_lon)]
        
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            first_edge = list(edge_data.values())[0]
            if isinstance(first_edge, dict) and 'geometry' in first_edge:
                # If geometry is a shapely LineString
                coords = list(first_edge['geometry'].coords)
                # coords are (lon, lat), so swap to (lat, lon)
                segment_coords = [(lat, lon) for lon, lat in coords]
        
        # Avoid duplicating the connection point
        if full_route_coords and full_route_coords[-1] == segment_coords[0]:
            full_route_coords.extend(segment_coords[1:])
        else:
            full_route_coords.extend(segment_coords)
    
    # Center the map on the midpoint of the route
    lats = [pt[0] for pt in full_route_coords]
    lons = [pt[1] for pt in full_route_coords]
    center = (np.mean(lats), np.mean(lons))
    
    # Create a Folium map
    m = folium.Map(location=center, zoom_start=14, tiles='cartodbpositron')
    
    # Draw the route
    folium.PolyLine(full_route_coords, color="blue", weight=5, opacity=0.8).add_to(m)
    
    # Mark origin and destination
    folium.Marker(full_route_coords[0], popup="Origin", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(full_route_coords[-1], popup="Destination", icon=folium.Icon(color="red")).add_to(m)
    
    # Add title
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

###############################################################################
# 7. Main Function: end-to-end pipeline
###############################################################################
def main():
    # Path to your CSV file
    csv_path = "University_Dataset(tmc_raw_data_2020_2029).csv"
    
    print("1) Processing traffic data...")
    data = process_data(csv_path)
    
    # Choose a time interval (for demonstration)
    intervals = {"1": (7, 9), "2": (11, 14), "3": (16, 19)}
    print("\nSelect a time interval:")
    print("1. 07:00-09:00")
    print("2. 11:00-14:00")
    print("3. 16:00-19:00")
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    time_interval = intervals.get(choice, (7, 9))
    start_hr, end_hr = time_interval
    print(f"Using time interval: {start_hr:02d}:00 - {end_hr:02d}:00")
    
    # Filter data for that time interval
    if 'start_time' in data.columns:
        period_data = data[data['start_time'].dt.hour.between(start_hr, end_hr)]
    else:
        period_data = data  # If there's no 'start_time' column, use all data
    
    if period_data.empty:
        print("No data found for the selected time interval. Exiting.")
        return
    
    print("\n2) Downloading road network...")
    G = load_osm_graph_bbox()
    
    print("3) Mapping traffic data to road nodes...")
    node_congestion = map_congestion_to_nodes(period_data, G)
    G = update_graph_congestion(G, node_congestion)
    
    print("4) Updating edge weights with congestion (alpha=0.7)...")
    G = update_edge_weights(G, alpha=0.7)
    
    print("\nEnter origin coordinates (decimal degrees):")
    origin_lat = float(input("Origin latitude: "))
    origin_lon = float(input("Origin longitude: "))
    
    print("\nEnter destination coordinates (decimal degrees):")
    dest_lat = float(input("Destination latitude: "))
    dest_lon = float(input("Destination longitude: "))
    
    # Snap origin/destination to the nearest nodes
    orig_node = ox.distance.nearest_nodes(G, origin_lon, origin_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lon, dest_lat)
    
    print(f"\nNearest origin node: {orig_node}")
    print(f"Nearest destination node: {dest_node}")
    
    print("5) Computing shortest path (neighbor-only) via Dijkstra...")
    route = compute_shortest_path(G, orig_node, dest_node)
    if route is None:
        print("No path found. Exiting.")
        return
    
    print("Route found:", route)
    
    print("\n6) Creating route map...")
    route_map = create_route_map(G, route, title=f"Optimal Route ({start_hr:02d}:00 - {end_hr:02d}:00)")
    if route_map:
        route_map.save("optimal_route.html")
        print("Map saved as optimal_route.html")
    else:
        print("Could not create a route map.")

if __name__ == "__main__":
    main()
