import networkx as nx
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance (in kilometers) between two points on the Earth.
    """
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def compute_edge_weight(lat1, lon1, lat2, lon2, congestion_index, alpha=0.5):
    """
    Combine the physical distance with a congestion factor.
    alpha is a tunable parameter controlling how much congestion affects the weight.
    """
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    weight = distance * (1 + alpha * (congestion_index / 100))
    return weight

def build_graph(data, time_interval, alpha=0.5):
    """
    Build a weighted graph from data filtered by the provided time_interval (tuple of start and end hour).
    Assumes data has 'start_time', 'latitude', 'longitude', and 'congestion_index'.
    Optionally, if available, uses 'location_name' for node identification.
    """
    start_hour, end_hour = time_interval
    period_data = data[data['start_time'].dt.hour.between(start_hour, end_hour)]
    
    if period_data.empty:
        print("No data available for the selected time interval!")
        return None
    
    # Group by location – if 'location_name' exists, use it; otherwise create one from latitude and longitude.
    if 'location_name' in period_data.columns:
        location_data = period_data.groupby(['location_name', 'latitude', 'longitude'])['congestion_index'].mean().reset_index()
    else:
        location_data = period_data.groupby(['latitude', 'longitude'])['congestion_index'].mean().reset_index()
        location_data['location_name'] = location_data.apply(lambda row: f"{row['latitude']}_{row['longitude']}", axis=1)
    
    G = nx.Graph()
    # Add nodes to the graph
    for _, row in location_data.iterrows():
        node_id = row['location_name']
        G.add_node(node_id, latitude=row['latitude'], longitude=row['longitude'], congestion=row['congestion_index'])
    
    # For demonstration, create edges between every pair of nodes.
    nodes = list(G.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            id1, attr1 = nodes[i]
            id2, attr2 = nodes[j]
            avg_congestion = (attr1['congestion'] + attr2['congestion']) / 2
            weight = compute_edge_weight(attr1['latitude'], attr1['longitude'],
                                         attr2['latitude'], attr2['longitude'],
                                         avg_congestion, alpha)
            G.add_edge(id1, id2, weight=weight, distance=haversine_distance(attr1['latitude'], attr1['longitude'],
                                                                           attr2['latitude'], attr2['longitude']))
    return G

def find_optimal_route(G, start_node, end_node):
    """
    Use Dijkstra's algorithm to find the optimal route based on the custom 'weight' attribute.
    """
    try:
        path = nx.dijkstra_path(G, source=start_node, target=end_node, weight='weight')
        return path
    except nx.NetworkXNoPath:
        print("No available path between the selected nodes.")
        return None
