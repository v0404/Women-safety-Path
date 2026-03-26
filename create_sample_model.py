import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a sample dataset
np.random.seed(42)
data = {
    'source': np.random.randint(0, 10, 1000),
    'destination': np.random.randint(0, 10, 1000),
    'time_of_day': np.random.randint(0, 3, 1000),  # 0: Morning, 1: Afternoon, 2: Evening, 3: Night
    'distance': np.random.uniform(1, 20, 1000),  # in km
    'population_density': np.random.uniform(1000, 10000, 1000),
    'streetlight_density': np.random.uniform(0, 100, 1000),
    'police_stations_nearby': np.random.randint(0, 5, 1000),
    'safety_score': np.random.randint(0, 100, 1000)
}

df = pd.DataFrame(data)

# Create safety labels based on safety score
df['safety_label'] = pd.cut(df['safety_score'], 
                           bins=[0, 30, 70, 100], 
                           labels=['Unsafe', 'Moderate', 'Safe'])

# Prepare features and target
X = df[['source', 'destination', 'time_of_day', 'distance', 'population_density', 
        'streetlight_density', 'police_stations_nearby']]
y = df['safety_label']

# Train a simple RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'safety_model.pkl')

print("Sample safety_model.pkl created successfully!")