import folium
import numpy as np

def create_route_map(G, route, title="Optimal Route"):
    """
    Creates an interactive map showing all nodes (gray markers), highlights the route (green markers),
    and draws a blue polyline along the route.
    """
    # Center the map on the average of all node coordinates
    lats = [data['latitude'] for _, data in G.nodes(data=True)]
    lons = [data['longitude'] for _, data in G.nodes(data=True)]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='cartodbpositron')
    
    # Add all nodes as gray markers
    for node, attr in G.nodes(data=True):
        folium.CircleMarker(
            location=[attr['latitude'], attr['longitude']],
            radius=4,
            color='gray',
            fill=True,
            fill_opacity=0.6,
            popup=node
        ).add_to(m)
    
    # Highlight the nodes along the optimal route
    for node in route:
        attr = G.nodes[node]
        folium.CircleMarker(
            location=[attr['latitude'], attr['longitude']],
            radius=6,
            color='green',
            fill=True,
            fill_opacity=1,
            popup=node
        ).add_to(m)
    
    # Draw the route as a polyline
    route_coords = [(G.nodes[node]['latitude'], G.nodes[node]['longitude']) for node in route]
    folium.PolyLine(route_coords, color='blue', weight=5, opacity=0.8).add_to(m)
    
    # Add a title overlay
    title_html = f'''
         <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
         '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m
