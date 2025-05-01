Optimized Ride Sharing and Trip Planning System 🚗

A Python-based web application that enables intelligent ride-sharing and efficient trip planning using advanced graph algorithms and geospatial mapping.

Features
1. Ride Sharing Optimization
Efficiently plans shared routes for 2 to 5 riders

Computes optimal routes based on distance and time

Calculates cost savings for shared rides

Interactive route visualization using OpenStreetMap

2. Multi-Stop Trip Planning
Optimizes travel routes with up to 5 stops

Two modes:

Fixed stop order

Dynamic TSP-based stop order

Estimates total trip distance, time, and cost

3. Graph Algorithm Implementations
Shortest Path: Dijkstra's (via BFS)

Network Coverage: Prim’s Minimum Spanning Tree (MST)

Route Optimization: Branch & Bound for Traveling Salesman Problem

Tech Stack
Frontend: Streamlit
Backend: Python 3.8+
Graph Algorithms: NetworkX
Mapping & Geocoding:
Folium
OpenStreetMap
OpenRouteService API

Setup and Installation
1. Clone the Repository
git clone https://github.com/yourusername/ride-share-optimizer.git
cd ride-share-optimizer

2. Install Dependencies
pip install -r requirements.txt

3. Add API Key
Sign up at openrouteservice.org and create a .env file:
ORS_API_KEY=your_open_route_service_api_key

4. Run the Application
streamlit run app.py

🔍Usage

 Ride Sharing

Add 2 to 5 pickup and drop-off locations

Choose transportation mode (Car/Bike)

View shared route, time saved, and cost saved

🧭 Multi-Stop Trip Planner
Add 3 to 5 stops

Choose between:

Fixed order

Optimized order (TSP)

Visualize optimal path and calculate total cost

📊 Output
Interactive map with route
Distance, duration, and estimated fuel/cost
Optimal rider grouping (for sharing mode)
