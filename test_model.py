import joblib
import numpy as np

# Load the model and landmarks
model = joblib.load('safety_model.pkl')
landmarks = joblib.load('landmarks.pkl')

print("Model loaded successfully!")
print("Landmarks:", landmarks)

# Test prediction
# Create sample features for prediction
source_idx = 0  # First landmark
destination_idx = 1  # Second landmark
time_of_day = 3  # Night
distance = 5.0  # km
population_density = 5000
streetlight_density = 70
police_stations_nearby = 2

# Create feature array
features = np.array([[source_idx, destination_idx, time_of_day, distance, 
                     population_density, streetlight_density, police_stations_nearby]])

print("Features shape:", features.shape)

# Predict safety label
prediction = model.predict(features)[0]
print("Prediction:", prediction)

# Get prediction probabilities
probabilities = model.predict_proba(features)[0]
print("Probabilities:", probabilities)