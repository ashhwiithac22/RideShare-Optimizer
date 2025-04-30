import streamlit as st
import networkx as nx
import folium
from folium.features import DivIcon
from streamlit_folium import folium_static
import pandas as pd
from geopy.distance import geodesic
import heapq
from collections import deque
import geocoder
import time
import random
import requests
import polyline
import os
import math
st.set_page_config(page_title="Optimized Ride Sharing System", layout="wide")
ORS_API_KEY = "5b3ce3597851110001cf6248f28aace8bda64b72b2d6a3fd3e42a943"  
if "graph" not in st.session_state:
    st.session_state.graph = nx.Graph()
if "locations" not in st.session_state: #geocoded_locations
    st.session_state.locations = {}
if "map_center" not in st.session_state: #sets India as default map_center
    st.session_state.map_center = [20.5937, 78.9629]  # Default India coordinates

# Geocoding (converts place names into coordinates)
def geocode_location(location_name):
    """Gets coordinates for a location """
    if location_name in st.session_state.locations:
        return st.session_state.locations[location_name]
    
    try:
        #if location not added tries to get coordinates using openstreetmap
        g = geocoder.osm(f"{location_name}, India")
        if g.ok:#if geocoder is ok returns the successful coordinates
            coords = (g.lat, g.lng)
            st.session_state.locations[location_name] = coords
            st.session_state.map_center = [g.lat, g.lng]
            return coords
        else:
            # Fallback to OpenRouteService Geocoding, 
            #when we the request to ors the API tells to return data in JSON format which can be easily processed by python
            headers = {
                'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
            }
            params = {
                'api_key': ORS_API_KEY,
                'text': f"{location_name}, India",
                'size': 1
            }
            response = requests.get('https://api.openrouteservice.org/geocode/search', 
                                  headers=headers, params=params)
            
            if response.status_code == 200: #status = 200 means server responded correctly 
                data = response.json()
                if data['features']:
                    coords = data['features'][0]['geometry']['coordinates']
                    coords = (coords[1], coords[0])  # Convert to (lat, lng)
                    st.session_state.locations[location_name] = coords
                    st.session_state.map_center = [coords[0], coords[1]]
                    return coords
            
            # Final fallback - if both method fails generate random coordinates in India
            st.warning(f"Could not find exact coordinates for '{location_name}'. Using approximate coordinates.")
            lat = 20.5937 + (random.random() - 0.5) * 10  # Random within India
            lng = 78.9629 + (random.random() - 0.5) * 10
            coords = (lat, lng)
            st.session_state.locations[location_name] = coords
            return coords
            
    except Exception as e: #if there is any error 
        st.error(f"Geocoding error: {str(e)}. Using default coordinates.")
        lat = 20.5937 + (random.random() - 0.5) * 10
        lng = 78.9629 + (random.random() - 0.5) * 10
        coords = (lat, lng)
        st.session_state.locations[location_name] = coords
        return coords

# Get route from OpenRouteService
def get_ors_route(start_coords, end_coords, profile='driving-car'):
    """Get route details from OpenRouteService API"""
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    body = {
        "coordinates": [
            [start_coords[1], start_coords[0]],  # ORS expects [lng, lat]
            [end_coords[1], end_coords[0]]
        ],
        "instructions": "false",
        "preference": "fastest"
    }
    
    try:
        response = requests.post(
            f'https://api.openrouteservice.org/v2/directions/{profile}/geojson',
            headers=headers,
            json=body
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['features']:
                feature = data['features'][0]
                distance = feature['properties']['segments'][0]['distance'] / 1000  # Convert to km
                duration = feature['properties']['segments'][0]['duration'] / 60  # Convert to minutes
                
                # Cost calculation based on profile
                if profile == 'driving-car':
                    base_fare = 50  # Base fare in rupees
                    per_km_rate = 15  # Rate per km in rupees
                    cost = base_fare + (distance * per_km_rate)
                else:  # driving-bike
                    base_fare = 20  # Base fare in rupees
                    per_km_rate = 8  # Rate per km in rupees
                    cost = base_fare + (distance * per_km_rate)
                
                # Decode polyline geometry
                geometry = feature['geometry']['coordinates']
                # Convert to lat,lng format
                geometry = [[coord[1], coord[0]] for coord in geometry]
                
                return {
                    "distance": distance,
                    "time": duration,
                    "cost": cost,
                    "geometry": geometry
                }
    
    except Exception as e:
        st.error(f"Routing API error: {str(e)}")
    
    # Fallback calculation if API fails
    direct_distance = geodesic(start_coords, end_coords).km
    road_factor = 1.3
    road_distance = direct_distance * road_factor
    avg_speed = 30  # km/h
    travel_time = (road_distance / avg_speed) * 60
    
    if profile == 'driving-car':
        base_fare = 50
        per_km_rate = 15
    else:  # driving-bike
        base_fare = 20
        per_km_rate = 8
    
    cost = base_fare + (road_distance * per_km_rate)
    
    return {
        "distance": road_distance,
        "time": travel_time,
        "cost": cost,
        "geometry": [
            [start_coords[1], start_coords[0]],
            [end_coords[1], end_coords[0]]
        ]
    }

# Route calculation with OpenRouteService
def calculate_route(source_coords, dest_coords, profile='driving-car'):
    """Calculate route with accurate metrics"""
    route_data = get_ors_route(source_coords, dest_coords, profile)
    
    return {
        "distance": route_data["distance"],
        "time": route_data["time"],
        "cost": route_data["cost"],
        "geometry": route_data["geometry"],
        "profile": profile
    }

# Graph algorithms - BFS for shortest path finding between 2 places
#here distance is called as weight
def bfs_shortest_path(graph, source, target, weight_attr='weight'):
    if source not in graph or target not in graph:
        return None, float('inf')
    
     #priority queue implementation 
    queue = []
    heapq.heappush(queue, (0, [source]))
    visited = set()
    
    while queue:
        current_weight, path = heapq.heappop(queue)
        current_node = path[-1]
        
        if current_node == target: #checks if destination has reached
            return path, current_weight
            
        if current_node in visited:
            continue
            
        visited.add(current_node) #keeps track of already visited places
        
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                edge_weight = graph[current_node][neighbor].get(weight_attr, 1)
                new_weight = current_weight + edge_weight
                new_path = path + [neighbor]
                heapq.heappush(queue, (new_weight, new_path))
    
    return None, float('inf')

# Prim's algorithm for MST 
#to find least total distance and time without any cycles
def prims_minimum_spanning_tree(graph, weight_attr='weight'):
    #returns empty graph if no nodes exists
    if not graph.nodes:
        return nx.Graph()
    
    mst = nx.Graph()
    start_node = list(graph.nodes)[0]
    mst.add_node(start_node)
    
    edges = []
    for u, v, data in graph.edges(data=True):
        if u == start_node or v == start_node:
            if weight_attr in data:
                edges.append((data[weight_attr], u, v))
    
    heapq.heapify(edges) #converts edges to minimum heap using heapq , always picks the lowest distance or cost cost
    
    while edges and len(mst.nodes) < len(graph.nodes):
        weight, u, v = heapq.heappop(edges)
        
        if (u in mst.nodes) != (v in mst.nodes): #expands the mst by not forming the loop
            if u in mst.nodes:
                mst.add_node(v)
                mst.add_edge(u, v, **graph[u][v])
                node_to_expand = v
            else:
                mst.add_node(u)
                mst.add_edge(u, v, **graph[u][v])
                node_to_expand = u
                
            for neighbor in graph.neighbors(node_to_expand):
                if neighbor not in mst.nodes:
                    if weight_attr in graph[node_to_expand][neighbor]:
                        heapq.heappush(edges, (graph[node_to_expand][neighbor][weight_attr], 
                                      node_to_expand, neighbor))
    
    return mst

# TSP using Branch and Bound,tries all possibilities and skips bad ones early 
class TSPSolver:
    def __init__(self, graph, weight_attr='weight'):
        self.graph = graph
        self.weight_attr = weight_attr
        self.n = len(graph.nodes) #number of nodes
        self.final_path = []#best path found so far 
        self.final_res = float('inf') #initial minimum cost is equal to infinity
        self.nodes = list(graph.nodes)
        self.node_indices = {node: idx for idx, node in enumerate(self.nodes)}
        
    def copy_to_final(self, curr_path): #copies the best path,when a better path is found stores it as final path 
        self.final_path = curr_path.copy()
        
    def first_min(self, i, adj): #finds the first minimum edge
        min_val = float('inf')
        for k in range(self.n):
            if adj[i][k] < min_val and i != k:
                min_val = adj[i][k]
        return min_val
        
    def second_min(self, i, adj):#finds the second minimum edge from first minimum edge
        first, second = float('inf'), float('inf')
        for j in range(self.n):
            if i == j:
                continue
            if adj[i][j] <= first:
                second = first
                first = adj[i][j]
            elif adj[i][j] <= second and adj[i][j] != first: #ignores self loop
                second = adj[i][j]
        return second
        
    def tsp_rec(self, adj, curr_bound, curr_weight, level, curr_path, visited):
        if level == self.n:
            if adj[curr_path[level-1]][curr_path[0]] != 0:
                curr_res = curr_weight + adj[curr_path[level-1]][curr_path[0]]
                if curr_res < self.final_res:
                    self.copy_to_final(curr_path)
                    self.final_res = curr_res
            return
            
        for i in range(self.n):
            if adj[curr_path[level-1]][i] != 0 and not visited[i]:
                temp = curr_bound #lower bound(curr bound) sum of two minimum edges
                curr_weight += adj[curr_path[level-1]][i] #actual path cost so far 
                
                if level == 1:
                    new_bound = self.first_min(curr_path[level-1], adj) + self.first_min(i, adj) #second minimum from previous city and new edge
                else:
                    new_bound = curr_bound - self.second_min(curr_path[level-1], adj) + adj[curr_path[level-1]][i]
                
                if new_bound + curr_weight < self.final_res:
                    curr_path[level] = i
                    visited[i] = True
                    self.tsp_rec(adj, new_bound, curr_weight, level+1, curr_path, visited)
                
                curr_weight -= adj[curr_path[level-1]][i] #Explore further only if this new path could lead to a better solution 
                curr_bound = temp
                visited = [False] * self.n
                for j in range(level):
                    if curr_path[j] != -1:
                        visited[curr_path[j]] = True
                        
    def solve(self, start_node):
        if self.n == 0:
            return [], 0
            
        # Create adjacency  matrix(distance)
        adj = [[0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    _, dist = bfs_shortest_path(self.graph, self.nodes[i], self.nodes[j], self.weight_attr)
                    adj[i][j] = dist if dist != float('inf') else 0
                    
        curr_path = [-1] * (self.n + 1)
        visited = [False] * self.n
        
        start_idx = self.node_indices[start_node]
        curr_path[0] = start_idx
        visited[start_idx] = True
        
        curr_bound = 0
        for i in range(self.n):
            curr_bound += (self.first_min(i, adj) + self.second_min(i, adj))
        curr_bound = math.ceil(curr_bound / 2)
        
        self.tsp_rec(adj, curr_bound, 0, 1, curr_path, visited)
        
        # Convert indices back to node names
        final_path_nodes = [self.nodes[i] for i in self.final_path]
        return final_path_nodes, self.final_res

def find_path_cost(graph, path, attr):
    if not path or len(path) < 2:
        return 0
        
    total = 0
    for i in range(len(path) - 1):
        if path[i] in graph and path[i+1] in graph[path[i]] and attr in graph[path[i]][path[i+1]]:
            total += graph[path[i]][path[i+1]][attr]
    return total

#finds best pickup order to minimize distance. Used when riders have different starting points.
def find_optimal_pickup_order(graph, rider_origins, destination, weight_attr='distance'):
    if len(rider_origins) == 1:
        return rider_origins
    
    #picks the nearest rider using bfs
    best_order = []
    best_distance = float('inf')

    for start_rider in rider_origins:
        current_order = [start_rider]
        remaining = [r for r in rider_origins if r != start_rider]
        current_location = start_rider
        total_distance = 0
        while remaining:
            next_rider = None
            next_distance = float('inf')
            
            for rider in remaining:
                path, dist = bfs_shortest_path(graph, current_location, rider, weight_attr)
                if path and dist < next_distance:
                    next_distance = dist
                    next_rider = rider
            
            if next_rider:
                current_order.append(next_rider)
                remaining.remove(next_rider)
                current_location = next_rider
                total_distance += next_distance
            else:
                break
        
        # Calculate distance to destination
        final_path, final_dist = bfs_shortest_path(graph, current_location, destination, weight_attr)
        if final_path:
            total_distance += final_dist
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_order = current_order
    
    return best_order

def find_shared_route(graph, rider_info, weight_attr='distance'):
    #Find optimal shared route for multiple riders using BFS
    origins = [rider["origin"] for rider in rider_info]
    destinations = [rider["destination"] for rider in rider_info]
    
    # If all riders have the same origin
    if len(set(origins)) == 1:
        pickup_order = origins[:1]  # Just one pickup needed
    else:
        # Find furthest destination
        max_dist = 0
        furthest_dest = destinations[0]
        
        for i, dest1 in enumerate(destinations):
            for dest2 in destinations[i+1:]:
                path, dist = bfs_shortest_path(graph, dest1, dest2, weight_attr)
                if path and dist > max_dist:
                    max_dist = dist
                    furthest_dest = dest1 if dist > 0 else dest2
        
        # Find optimal pickup order
        pickup_order = find_optimal_pickup_order(graph, origins, furthest_dest, weight_attr)
    
    # Find optimal drop-off order
    if len(set(destinations)) == 1:
        dropoff_order = destinations[:1]  # All go to same place
    else:
        current_location = pickup_order[-1]  # Last pickup point
        dropoff_order = []
        remaining_dests = destinations.copy()
        
        while remaining_dests:
            next_dest = None
            next_distance = float('inf')
            
            for dest in remaining_dests:
                path, dist = bfs_shortest_path(graph, current_location, dest, weight_attr)
                if path and dist < next_distance:
                    next_distance = dist
                    next_dest = dest
            
            if next_dest:
                dropoff_order.append(next_dest)
                remaining_dests.remove(next_dest)
                current_location = next_dest
            else:
                break
    
    # Calculating total route
    route_points = []
    current_point = pickup_order[0]
    route_points.append(current_point)
    
    # Add pickup points
    for point in pickup_order[1:]:
        if point != current_point:
            path, _ = bfs_shortest_path(graph, current_point, point, weight_attr)
            if path:
                route_points.extend(path[1:])  
                current_point = point
    
    # Add dropoff points
    for point in dropoff_order:
        if point != current_point:
            path, _ = bfs_shortest_path(graph, current_point, point, weight_attr)
            if path:
                route_points.extend(path[1:])  
                current_point = point
    
    return route_points

# UI Components
st.title("🚗 Optimized Ride Sharing System")

# Sidebar for location 
with st.sidebar:
    st.header("📍 Manage Locations")
    
    # Input method selection
    input_method = st.radio("Location Input Method", 
                           ["Enter Address", "Select from Examples"], 
                           index=0,
                           help="Enter your own location or choose from examples")
    
    if input_method == "Enter Address":
        new_location = st.text_input("Enter Location", 
                                    placeholder="e.g., Mall, University, Landmark, Address",
                                    help="Enter specific location name with city if possible")
        location_info = st.text_input("Additional Details (Optional)", 
                                     placeholder="e.g., City, State",
                                     help="Add city, state or other details to improve geocoding accuracy")
        
        if location_info and not location_info in new_location:
            full_location = f"{new_location}, {location_info}"
        else:
            full_location = new_location
            
        if st.button("Add Location"):
            if full_location:
                try:
                    with st.spinner("Finding location..."):
                        coords = geocode_location(full_location)
                        if coords:
                            st.session_state.graph.add_node(full_location)
                            st.success(f"Added {full_location} at coordinates {coords}")
                        else:
                            st.error(f"Could not find coordinates for {full_location}")
                except Exception as e:
                    st.error(f"Failed to add location: {str(e)}")
    else:
        example_locations = [
            "Marina Beach, Chennai",
            "Gateway of India, Mumbai",
            "Hawa Mahal, Jaipur",
            "Charminar, Hyderabad",
            "Victoria Memorial, Kolkata",
            "Lalbagh Botanical Garden, Bangalore",
            "Golden Temple, Amritsar",
            "Taj Mahal, Agra",
            "India Gate, New Delhi",
            "Meenakshi Temple, Madurai"
        ]
        
        selected_location = st.selectbox("Choose Location", example_locations)
        
        if st.button("Add Selected Location"):
            if selected_location:
                try:
                    with st.spinner("Finding location..."):
                        coords = geocode_location(selected_location)
                        if coords:
                            st.session_state.graph.add_node(selected_location)
                            st.success(f"Added {selected_location} at coordinates {coords}")
                        else:
                            st.error(f"Could not find coordinates for {selected_location}")
                except Exception as e:
                    st.error(f"Failed to add location: {str(e)}")

    #creating connections between locations
    st.header("🔗 Create Connections")
    
    connection_method = st.radio("Connection Method", 
                               ["Manual Connections", "Auto-connect All"], 
                               index=0,
                               help="Create connections manually or connect all locations automatically")
    
    transport_mode = st.radio("Transport Mode", 
                            ["Car", "Bike"], 
                            index=0,
                            help="Select mode of transport for route calculations")
    
    profile = 'driving-car' if transport_mode == "Car" else 'driving-bike'
    
    if connection_method == "Manual Connections":
        if len(st.session_state.graph.nodes) >= 2:
            loc1 = st.selectbox("From", list(st.session_state.graph.nodes), key="from")
            loc2 = st.selectbox("To", list(st.session_state.graph.nodes), key="to")
            
            is_bidirectional = st.checkbox("Bidirectional Connection", value=True, 
                                         help="Create connection in both directions")
            
            if st.button("Create Connection"):
                if loc1 == loc2:
                    st.error("Source and destination must be different")
                else:
                    try:
                        with st.spinner("Creating connection..."):
                            source_coords = st.session_state.locations[loc1]
                            dest_coords = st.session_state.locations[loc2]
                            
                            route_data = calculate_route(source_coords, dest_coords, profile)
                            
                            st.session_state.graph.add_edge(
                                loc1, loc2,
                                weight=route_data["distance"],
                                distance=route_data["distance"],
                                time=route_data["time"],
                                cost=route_data["cost"],
                                geometry=route_data["geometry"],
                                profile=profile
                            )
                            
                            if is_bidirectional:
                                st.session_state.graph.add_edge(
                                    loc2, loc1,
                                    weight=route_data["distance"],
                                    distance=route_data["distance"],
                                    time=route_data["time"],
                                    cost=route_data["cost"],
                                    geometry=route_data["geometry"][::-1],
                                    profile=profile
                                )
                                st.success(f"Added bidirectional connection: {route_data['distance']:.2f} km")
                            else:
                                st.success(f"Added one-way connection: {route_data['distance']:.2f} km")
                    except Exception as e:
                        st.error(f"Failed to add connection: {str(e)}")
        else:
            st.info("Add at least 2 locations to create connections")
    else:
        # Auto connect all
        if len(st.session_state.graph.nodes) >= 2:
            if st.button("Connect All Locations"):
                try:
                    with st.spinner("Creating connections between all locations..."):
                        locations = list(st.session_state.graph.nodes)
                        count = 0
                        
                        for i, loc1 in enumerate(locations):
                            for loc2 in locations[i+1:]:
                                source_coords = st.session_state.locations[loc1]
                                dest_coords = st.session_state.locations[loc2]
                                
                                route_data = calculate_route(source_coords, dest_coords, profile)
                                
                                # Add bidirectional edges
                                st.session_state.graph.add_edge(
                                    loc1, loc2,
                                    weight=route_data["distance"],
                                    distance=route_data["distance"],
                                    time=route_data["time"],
                                    cost=route_data["cost"],
                                    geometry=route_data["geometry"],
                                    profile=profile
                                )
                                
                                st.session_state.graph.add_edge(
                                    loc2, loc1,
                                    weight=route_data["distance"],
                                    distance=route_data["distance"],
                                    time=route_data["time"],
                                    cost=route_data["cost"],
                                    geometry=route_data["geometry"][::-1],
                                    profile=profile
                                )
                                
                                count += 1
                        
                        st.success(f"Created {count} bidirectional connections between all locations")
                except Exception as e:
                    st.error(f"Failed to connect locations: {str(e)}")
        else:
            st.info("Add at least 2 locations to create connections")
    with st.expander("Network Statistics", expanded=True):
        if st.session_state.graph.nodes:
            num_nodes = len(st.session_state.graph.nodes)
            num_edges = len(st.session_state.graph.edges)
            
            st.metric("Total Locations", num_nodes)
            st.metric("Total Connections", num_edges)
            
            if num_nodes > 1:
                try:
                    density = nx.density(st.session_state.graph)
                    st.metric("Network Density", f"{density:.2f}")
                    components = list(nx.connected_components(st.session_state.graph))
                    st.write(f"Connected Components: {len(components)}")
                    
                    # Only calculate centrality if there are edges
                    if num_edges > 0:
                        try:
                            betweenness = nx.betweenness_centrality(st.session_state.graph, weight="weight")
                            if betweenness:
                                most_central = max(betweenness, key=betweenness.get)
                                st.write(f"Most Central Hub: {most_central}")
                        except:
                            st.write("Could not calculate centrality metrics")
                except Exception as e:
                    st.error(f"Could not calculate statistics: {str(e)}")
        else:
            st.info("Add locations to see network statistics")

    with st.expander("Save/Load Network", expanded=False):
        if st.button("Clear Network"):
            st.session_state.graph = nx.Graph()
            st.session_state.locations = {}
            st.success("Network cleared")

# Main content
if st.session_state.graph.nodes:
    st.subheader("Current Network")
    
    # Create map with all locations and connections
    m = folium.Map(location=st.session_state.map_center, zoom_start=5 if len(st.session_state.graph.nodes) > 3 else 12)
    
    # Add markers for all locations
    for loc in st.session_state.graph.nodes:
        if loc in st.session_state.locations:
            coords = st.session_state.locations[loc]
            folium.Marker(
                coords,
                popup=loc,
                tooltip=loc,
                icon=folium.Icon(color="blue")
            ).add_to(m)
    
    # Add edges for all connections
    for u, v, data in st.session_state.graph.edges(data=True):
        if "geometry" in data:
            line_color = "blue" if data.get("profile", "") == "driving-car" else "green"
            folium.PolyLine(
                data["geometry"],
                color=line_color,
                weight=2,
                opacity=0.7,
                tooltip=f"{u} to {v}: {data.get('distance', 0):.2f} km"
            ).add_to(m)
    
    # Display the map
    st.write("Network Visualization:")
    folium_static(m, width=700)

# Ride Sharing Optimization
st.header("👥 Ride Sharing Optimization")

if len(st.session_state.graph.nodes) >= 2:
    st.subheader("Find Optimal Shared Rides")
    
    # Number of riders selection
    num_riders = st.radio("Number of riders", [2, 3, 4, 5], index=0, horizontal=True)
    
    # Rider information
    rider_info = []
    cols = st.columns(min(num_riders, 3)) 
    
    for i in range(num_riders):
        with cols[i % len(cols)]:
            st.markdown(f"### Rider {i+1}")
            rider_origin = st.selectbox(f"Origin for Rider {i+1}", 
                                      list(st.session_state.graph.nodes), 
                                      key=f"rider_{i}_origin")
            rider_dest = st.selectbox(f"Destination for Rider {i+1}", 
                                     list(st.session_state.graph.nodes), 
                                     key=f"rider_{i}_dest")
            
            rider_info.append({
                "id": i+1,
                "origin": rider_origin,
                "destination": rider_dest
            })
    
    optimize_type = st.radio("Optimization Priority", 
                           ["Distance", "Time", "Cost"], 
                           horizontal=True,
                           index=0,
                           key="optimize_type_radio")
    
    weight_mapping = {
        "Distance": "distance",
        "Time": "time",
        "Cost": "cost"
    }
    
    weight_attr = weight_mapping[optimize_type]
    
    if st.button("Find Optimal Shared Rides"):
        try:
            with st.spinner("Finding optimal ride sharing plan..."):
                # Verify that the graph is connected and paths exist
                connected = nx.is_connected(st.session_state.graph.to_undirected())
                
                if not connected:
                    st.error("The network is not fully connected. Please add more connections between locations.")
                else:
                    # Check if paths exist for all riders
                    valid_paths = True
                    for rider in rider_info:
                        if not nx.has_path(st.session_state.graph, rider["origin"], rider["destination"]):
                            st.error(f"No path exists for Rider {rider['id']} from {rider['origin']} to {rider['destination']}")
                            valid_paths = False
                    
                    if valid_paths:
                        # Calculate individual routes
                        individual_routes = []
                        total_individual = {
                            "distance": 0,
                            "time": 0,
                            "cost": 0
                        }
                        
                        for rider in rider_info:
                            path, dist = bfs_shortest_path(
                                st.session_state.graph, 
                                rider["origin"], 
                                rider["destination"], 
                                weight_attr
                            )
                            
                            if path:
                                time = find_path_cost(st.session_state.graph, path, "time")
                                cost = find_path_cost(st.session_state.graph, path, "cost")
                                distance = find_path_cost(st.session_state.graph, path, "distance")
                                
                                individual_routes.append({
                                    "rider_id": rider["id"],
                                    "path": path,
                                    "distance": distance,
                                    "time": time,
                                    "cost": cost
                                })
                                
                                total_individual["distance"] += distance
                                total_individual["time"] += time
                                total_individual["cost"] += cost
                        
                        # Find shared route
                        shared_route = find_shared_route(st.session_state.graph, rider_info, weight_attr)
                        
                        # Calculate metrics for shared route
                        shared_distance = 0
                        shared_time = 0
                        shared_cost = 0
                        
                        for i in range(len(shared_route) - 1):
                            u = shared_route[i]
                            v = shared_route[i + 1]
                            
                            if v in st.session_state.graph[u]:
                                edge_data = st.session_state.graph[u][v]
                                shared_distance += edge_data.get("distance", 0)
                                shared_time += edge_data.get("time", 0)
                                shared_cost += edge_data.get("cost", 0)
                        
                        # Apply ride sharing discount
                        # The more riders, the bigger the discount
                        discount_factors = {2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}
                        discount_factor = discount_factors.get(num_riders, 0.8)
                        shared_cost *= discount_factor
                        
                        # Calculating savings
                        savings = {
                            "distance": total_individual["distance"] - shared_distance,
                            "time": total_individual["time"] - shared_time,
                            "cost": total_individual["cost"] - shared_cost
                        }
                        
                        # Displaying results
                        st.success("Ride Sharing Plan Optimized!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Individual Routes")
                            st.write(f"Total Distance: {total_individual['distance']:.2f} km")
                            st.write(f"Total Time: {total_individual['time']:.1f} mins")
                            st.write(f"Total Cost: ₹{total_individual['cost']:.0f}")
                        
                        with col2:
                            st.subheader("Shared Route")
                            st.write(f"Total Distance: {shared_distance:.2f} km")
                            st.write(f"Total Time: {shared_time:.1f} mins")
                            st.write(f"Total Cost: ₹{shared_cost:.0f}")
                        
                        st.subheader("Savings")
                        st.write(f"Distance Saved: {savings['distance']:.2f} km")
                        st.write(f"Time Saved: {savings['time']:.1f} mins")
                        st.write(f"Cost Savings: ₹{savings['cost']:.0f}")
                        
                        # Calculate per-rider cost in shared model
                        per_rider_cost = shared_cost / num_riders
                        st.write(f"Cost per rider: ₹{per_rider_cost:.0f}")
                        
                        # Show route on map
                        m = folium.Map(location=st.session_state.map_center, zoom_start=12)
                        
                        # Add markers for all rider origins
                        for rider in rider_info:
                            origin_coords = st.session_state.locations[rider["origin"]]
                            dest_coords = st.session_state.locations[rider["destination"]]
                            
                            folium.Marker(
                                origin_coords,
                                popup=f"Rider {rider['id']} Origin: {rider['origin']}",
                                tooltip=f"Rider {rider['id']} Origin",
                                icon=folium.Icon(color="blue", icon="user")
                            ).add_to(m)
                            
                            folium.Marker(
                                dest_coords,
                                popup=f"Rider {rider['id']} Destination: {rider['destination']}",
                                tooltip=f"Rider {rider['id']} Destination",
                                icon=folium.Icon(color="red", icon="flag")
                            ).add_to(m)
                        
                        # Add shared route on the map
                        for i in range(len(shared_route) - 1):
                            u = shared_route[i]
                            v = shared_route[i + 1]
                            
                            if v in st.session_state.graph[u] and "geometry" in st.session_state.graph[u][v]:
                                line_color = "blue" if st.session_state.graph[u][v].get("profile", "") == "driving-car" else "green"
                                folium.PolyLine(
                                    st.session_state.graph[u][v]["geometry"],
                                    color=line_color,
                                    weight=4,
                                    opacity=0.8,
                                    tooltip=f"{u} → {v}"
                                ).add_to(m)
                        
                        # Add route markers with sequence numbers
                        route_points = shared_route
                        for i, point in enumerate(route_points):
                            if point in st.session_state.locations:
                                coords = st.session_state.locations[point]
                                folium.CircleMarker(
                                    location=coords,
                                    radius=15,
                                    color="green",
                                    fill=True,
                                    fill_color="green",
                                    fill_opacity=0.7,
                                    tooltip=f"Stop {i+1}: {point}"
                                ).add_to(m)

                                folium.map.Marker(
                                    coords,
                                    icon=DivIcon(
                                        icon_size=(20, 20),
                                        icon_anchor=(10, 10),
                                        html=f'<div style="font-size: 10pt; color: white; text-align: center; font-weight: bold;">{i+1}</div>',
                                    )
                                ).add_to(m)
                        
                        folium_static(m, width=700)
                        
                        # Display detailed route information
                        st.subheader("Detailed Route")
                        route_details = []
                        for i in range(len(shared_route) - 1):
                            u = shared_route[i]
                            v = shared_route[i + 1]
                            if v in st.session_state.graph[u]:
                                edge_data = st.session_state.graph[u][v]
                                action = ""
                                if u in [rider["origin"] for rider in rider_info]:
                                    rider_ids = [r["id"] for r in rider_info if r["origin"] == u]
                                    action = f"🚗 Pickup Rider {', '.join(map(str, rider_ids))}"
                                if v in [rider["destination"] for rider in rider_info]:
                                    rider_ids = [r["id"] for r in rider_info if r["destination"] == v]
                                    action = f"🏁 Dropoff Rider {', '.join(map(str, rider_ids))}"
                                
                                route_details.append({
                                    "Step": i + 1,
                                    "From": u,
                                    "To": v,
                                    "Distance": f"{edge_data.get('distance', 0):.2f} km",
                                    "Time": f"{edge_data.get('time', 0):.1f} mins",
                                    "Cost": f"₹{edge_data.get('cost', 0):.0f}",
                                    "Action": action
                                })
                        
                        route_df = pd.DataFrame(route_details)
                        st.dataframe(route_df, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Ride sharing optimization failed: {str(e)}")
            st.info("Try adding more connections between locations or check that all locations are connected")
else:
    st.info("Add at least 2 locations to use ride sharing optimization")

# Multi-Stop Trip Planning
st.header("🗺 Multi-Stop Trip Planning")

if len(st.session_state.graph.nodes) >= 2:
    st.subheader("Plan Your Trip")
    
    # Starting point
    start_point = st.selectbox("Starting Point", 
                              list(st.session_state.graph.nodes), 
                              key="trip_start")
    
    # Stops
    stops = []
    num_stops = st.slider("Number of stops", 1, 5, 2)
    
    for i in range(num_stops):
        stop = st.selectbox(f"Stop {i+1}", 
                           [loc for loc in st.session_state.graph.nodes if loc != start_point],
                           key=f"stop_{i}")
        stops.append(stop)
    
    # Trip optimization method
    trip_method = st.radio(
        "Trip Optimization Method",
        ["Shortest Path (Visit in given order)", "Optimal Route (TSP - Find best order)"],
        index=0,
        help="Choose whether to follow your stop order or let the system find the optimal order"
    )
    
    # Optimization metric
    trip_metric = st.radio(
        "Optimization Priority", 
        ["Distance", "Time", "Cost"], 
        horizontal=True,
        index=0,
        key="trip_metric_radio"
    )
    
    weight_attr = weight_mapping[trip_metric]
    
    if st.button("Plan Trip"):
        try:
            with st.spinner("Finding optimal route..."):
                # Verify that the graph is connected
                connected = nx.is_connected(st.session_state.graph.to_undirected())
                
                if not connected:
                    st.error("The network is not fully connected. Please add more connections between locations.")
                else:
                    # Check if paths exist between stops
                    valid_paths = True
                    for i in range(len(stops)):
                        prev = start_point if i == 0 else stops[i-1]
                        curr = stops[i]
                        if not nx.has_path(st.session_state.graph, prev, curr):
                            st.error(f"No path exists from {prev} to {curr}")
                            valid_paths = False
                            break
                    
                    if valid_paths:
                        # Building route based on selected method
                        if trip_method == "Shortest Path (Visit in given order)":
                            # Using the order provided by user
                            route = [start_point] + stops
                        else:
                            # Using TSP with Branch and Bound to find optimal order
                            tsp_solver = TSPSolver(st.session_state.graph, weight_attr)
                            route, _ = tsp_solver.solve(start_point)
                        
                        # Calculating detailed route segments
                        segments = []
                        total_distance = 0
                        total_time = 0
                        total_cost = 0
                        detailed_path = []
                        
                        for i in range(len(route) - 1):
                            source = route[i]
                            target = route[i + 1]
                            
                            path, dist = bfs_shortest_path(
                                st.session_state.graph, 
                                source, 
                                target, 
                                weight_attr
                            )
                            
                            if path:
                                time = find_path_cost(st.session_state.graph, path, "time")
                                cost = find_path_cost(st.session_state.graph, path, "cost")
                                distance = find_path_cost(st.session_state.graph, path, "distance")
                                
                                segments.append({
                                    "from": source,
                                    "to": target,
                                    "path": path,
                                    "distance": distance,
                                    "time": time,
                                    "cost": cost
                                })
                                
                                total_distance += distance
                                total_time += time
                                total_cost += cost
                                
                                # Adding to detailed path (for visualization)
                                if detailed_path and detailed_path[-1] == path[0]:
                                    detailed_path.extend(path[1:])
                                else:
                                    detailed_path.extend(path)
                        
                        if segments:
                            # Displaying results
                            st.success("Trip Planning Completed!")
                            st.subheader("Trip Summary")
                            st.write(f"Total Distance: {total_distance:.2f} km")
                            st.write(f"Total Time: {total_time:.1f} mins")
                            st.write(f"Total Cost: ₹{total_cost:.0f}")
                            
                            # Showing on map
                            m = folium.Map(location=st.session_state.map_center, zoom_start=12)
                            
                            # Adding markers for all stops
                            for i, point in enumerate(route):
                                coords = st.session_state.locations[point]
                                if i == 0:
                                    # Startinging point
                                    folium.Marker(
                                        coords,
                                        popup=f"Start: {point}",
                                        tooltip=f"Start: {point}",
                                        icon=folium.Icon(color="green", icon="home")
                                    ).add_to(m)
                                elif i == len(route) - 1:
                                    # Final destination
                                    folium.Marker(
                                        coords,
                                        popup=f"End: {point}",
                                        tooltip=f"End: {point}",
                                        icon=folium.Icon(color="red", icon="flag")
                                    ).add_to(m)
                                else:
                                    # Intermediate stop
                                    folium.Marker(
                                        coords,
                                        popup=f"Stop {i}: {point}",
                                        tooltip=f"Stop {i}: {point}",
                                        icon=folium.Icon(color="blue", icon="info-sign")
                                    ).add_to(m)
                                
                                # Adding stop number
                                folium.map.Marker(
                                    coords,
                                    icon=DivIcon(
                                        icon_size=(20, 20),
                                        icon_anchor=(12, 12),
                                        html=f'<div style="font-size: 10pt; background-color: white; border-radius: 10px; width : 20px; height : 20px; line-height: 20px; text-align: center; color: black; border: 2px solid {"green" if i == 0 else "red" if i == len(route) - 1 else "blue"}; font-weight: bold;">{i}</div>',
                                    )
                                ).add_to(m)
                            
                            # Adding route segments
                            for seg in segments:
                                path = seg["path"]
                                for i in range(len(path) - 1):
                                    u = path[i]
                                    v = path[i + 1]
                                    if v in st.session_state.graph[u] and "geometry" in st.session_state.graph[u][v]:
                                        line_color = "blue" if st.session_state.graph[u][v].get("profile", "") == "driving-car" else "green"
                                        folium.PolyLine(
                                            st.session_state.graph[u][v]["geometry"],
                                            color=line_color,
                                            weight=4,
                                            opacity=0.7,
                                            tooltip=f"{u} → {v}: {st.session_state.graph[u][v].get('distance', 0):.2f} km"
                                        ).add_to(m)
                            
                            folium_static(m, width=700)
                            
                            # Displaying detailed route information
                            st.subheader("Detailed Route")
                            route_details = []
                            for i, seg in enumerate(segments):
                                route_details.append({
                                    "Step": i + 1,
                                    "From": seg["from"],
                                    "To": seg["to"],
                                    "Distance": f"{seg['distance']:.2f} km",
                                    "Time": f"{seg['time']:.1f} mins",
                                    "Cost": f"₹{seg['cost']:.0f}"
                                })
                            
                            route_df = pd.DataFrame(route_details)
                            st.dataframe(route_df, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Trip planning failed: {str(e)}")
            st.info("Try adding more connections between locations or check that all locations are connected")
else:
    st.info("Add at least 2 locations to plan trips")

# Instructions for users
with st.expander("How to Use this System", expanded=False):
    st.markdown("""
    ## How to Use the Optimized Ride Sharing System

    ### Step 1: Add Locations 
    - Use the sidebar to add locations by either entering addresses or selecting from examples
    - For best results, add specific location names like "Gateway of India, Mumbai" rather than just "Gateway"
    - The more specific the location, the better the geocoding accuracy

    ### Step 2: Create Connections
    - Select transport mode (Car or Bike) - this affects route calculations
    - Create connections between locations either manually or automatically
    - Manual connections let you create specific links between places
    - Auto-connect creates connections between all locations

    ### Step 3: Use the Tools
    
    #### For Ride Sharing (2-5 riders):
    1. Select the number of riders (up to 5)
    2. Choose origin and destination for each rider
    3. Select optimization priority (Distance, Time, or Cost)
    4. Click "Find Optimal Shared Rides" to see the results
    
    #### For Multi-Stop Trip Planning:
    1. Select a starting point
    2. Choose how many stops you want (1-5)
    3. Select each stop location
    4. Choose whether to visit stops in your order or find optimal order
    5. Select optimization priority
    6. Click "Plan Trip" to see results

    ### Tips
    - Add at least 4-5 locations for meaningful ride sharing results
    - The system will automatically create efficient routes between locations
    - You can clear the network and start over using the "Clear Network" button
    """)