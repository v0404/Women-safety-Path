# Women Safety Prediction System

An AI-based system that predicts the safety score between two locations in Vijayawada using a trained RandomForest model. The system provides safety scores, alternate route suggestions, and visualizes routes on a map.

## Features

- Predict safety scores for routes in Vijayawada (0-100%)
- Categorize routes as Safe, Moderate, or Unsafe
- Suggest alternate safe routes when the direct route is unsafe
- Visualize routes on an interactive map
- Display nearby police stations
- Provide safety tips based on time of day

## Technologies Used

- **Streamlit**: Web application framework
- **Folium**: Interactive map visualization
- **Scikit-learn**: Machine learning model (RandomForest)
- **Geopy**: Geocoding services
- **Joblib**: Model serialization

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser

3. Select:
   - Source location
   - Destination location
   - Time of day (Morning/Afternoon/Evening/Night)

4. Click "Predict Safety" to get the results

## How It Works

1. The user selects source, destination, and time of day
2. The system uses a trained RandomForest model to predict the safety score
3. If the direct route is unsafe, an alternate safe route is suggested
4. Both routes are visualized on an interactive map
5. Safety tips are provided based on the time of day

## Model Information

The system uses a RandomForest classifier trained on features such as:
- Source and destination locations
- Time of day
- Distance between locations
- Population density
- Streetlight density
- Nearby police stations

## Map Features

- Red route: Direct route (color indicates safety level)
- Green route: Alternate safe route
- Blue markers: Police stations
- Green marker: Starting point
- Red marker: Destination

## Safety Tips

The system provides context-aware safety tips based on the time of day:
- Morning/Afternoon: General safety precautions
- Evening: Route safety recommendations
- Night: Enhanced safety measures