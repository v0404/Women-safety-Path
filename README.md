# AI-Based Women Safety Prediction System

This project aims to enhance women's safety by predicting the safety level of travel routes using machine learning techniques.

## Features

- Predicts route safety (Safe / Moderate / Unsafe)
- Generates safety score (0–100%)
- Machine Learning model (Random Forest)
- Interactive map visualization using Streamlit and Folium
- Suggests alternate safer routes dynamically
- Displays nearby police stations on map
- Considers time of day (Morning, Afternoon, Evening, Night)

## How It Works

The system takes:
- Source location
- Destination
- Time of travel

It then evaluates multiple features:
- Distance between locations
- Population density
- Streetlight density
- Number of nearby police stations

Using a trained ML model, it predicts:
- Safety label (Safe / Moderate / Unsafe)
- Safety score (%)

It also generates an alternate route passing through safer landmarks.

## Model Details

- Algorithm: Random Forest Classifier
- Features Used:
  - Source index
  - Destination index
  - Time of day
  - Distance
  - Population density
  - Streetlight density
  - Police stations nearby

## Technologies Used

- Python
- Streamlit (Frontend UI)
- Folium (Map visualization)
- Scikit-learn (Machine Learning)
- Pandas, NumPy

## Output

- Safety score (e.g., 72% Safe)
- Safety suggestions
- Route visualization (direct vs alternate)
- Police station markers

## Future Improvements

- Real-time crime data integration
- Live GPS tracking
- Traffic-aware route optimization
- Emergency alert system

---

Developed as a Machine Learning Project for academic purposes.
