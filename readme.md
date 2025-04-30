# Optimized Ride Sharing and Trip Planning System 🚗

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-FF4B4B)](https://streamlit.io/)
[![NetworkX](https://img.shields.io/badge/NetworkX-2.8.8-orange)](https://networkx.org/)

A graph-based optimization system that implements BFS (shortest path), Prim's (MST), and TSP algorithms to enable efficient ride-sharing and multi-stop trip planning.

## Key Features

- 🚘 **Ride Sharing Optimization**:
  - Finds optimal shared routes for 2-5 riders
  - Calculates distance, time, and cost savings
  - Visualizes shared routes on interactive maps

- 🗺 **Multi-Stop Trip Planning**:
  - Plans optimal routes with multiple stops
  - Two modes: Fixed order or TSP-optimized
  - Calculates total distance, time, and cost

- 📊 **Algorithm Implementations**:
  - BFS (Dijkstra's) for shortest path finding
  - Prim's algorithm for network analysis
  - Branch & Bound for Traveling Salesman Problem

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **Graph Algorithms**: NetworkX
- **Mapping**: Folium, OpenStreetMap
- **Geocoding/Routing**: OpenRouteService API

## Installation

1. Clone the repository:
git clone https://github.com/yourusername/ride-sharing-optimizer.git
cd ride-sharing-optimizer

Install dependencies:
pip install -r requirements.txt
Set up API key:
Get a free API key from OpenRouteService
Add it to the code where ORS_API_KEY is defined

Run the application:
streamlit run app.py

Use the sidebar to:
Add locations (manually or from examples)
Create connections between locations
Choose transport mode (Car/Bike)

Access main features:
Ride Sharing Optimization (2-5 riders)
Multi-Stop Trip Planning (up to 5 stops)

Algorithm Complexity
Algorithm	           Time Complexity	     Space Complexity
BFS (Dijkstra's)	 O(E + V log V)	             O(V)
Prim's MST         	O(E log V)	                O(V + E)
TSP (Branch & Bound)	O(n!)	                  O(n)
