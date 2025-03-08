import data_processing
import graph_module
import visualization

def main():
    # Path to the CSV dataset (make sure the file is in the project folder)
    csv_path = "University_Dataset(tmc_raw_data_2020_2029).csv"
    print("Processing data...")
    data = data_processing.process_data(csv_path)
    
    # Provide time interval options (in hours)
    intervals = {
        "1": (7, 9),
        "2": (11, 14),
        "3": (16, 19)
    }
    print("Select a time interval:")
    print("1. 07:00-09:00")
    print("2. 11:00-14:00")
    print("3. 16:00-19:00")
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    if choice not in intervals:
        print("Invalid choice. Defaulting to 07:00-09:00")
        time_interval = intervals["1"]
    else:
        time_interval = intervals[choice]
    
    print(f"Building graph for time interval {time_interval[0]}:00-{time_interval[1]}:00...")
    G = graph_module.build_graph(data, time_interval, alpha=0.7)
    if G is None:
        print("Graph could not be built. Exiting.")
        return
    
    # List available locations (nodes)
    nodes = list(G.nodes())
    if not nodes:
        print("No nodes available in the graph. Exiting.")
        return

    print("Available locations:")
    for idx, node in enumerate(nodes):
        print(f"{idx+1}. {node}")
    
    try:
        start_idx = int(input("Select start location (enter number): ")) - 1
        end_idx = int(input("Select end location (enter number): ")) - 1
    except ValueError:
        print("Invalid input. Exiting.")
        return
    
    if start_idx < 0 or start_idx >= len(nodes) or end_idx < 0 or end_idx >= len(nodes):
        print("Invalid location selection. Exiting.")
        return
    
    start_node = nodes[start_idx]
    end_node = nodes[end_idx]
    
    print(f"Calculating optimal route from {start_node} to {end_node}...")
    route = graph_module.find_optimal_route(G, start_node, end_node)
    if route is None:
        print("No route found. Exiting.")
        return
    
    print("Optimal route found:")
    for node in route:
        print(node)
    
    # Create the route map and save it as an HTML file
    print("Creating route map...")
    route_map = visualization.create_route_map(G, route, title=f"Optimal Route ({time_interval[0]:02d}:00-{time_interval[1]:02d}:00)")
    map_file = "optimal_route.html"
    route_map.save(map_file)
    print(f"Route map saved as {map_file}")

if __name__ == '__main__':
    main()
