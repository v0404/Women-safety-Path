import joblib
import os

# Test loading model with relative path
print("Current working directory:", os.getcwd())

try:
    print("Trying to load with relative path...")
    model = joblib.load('safety_model.pkl')
    landmarks = joblib.load('landmarks.pkl')
    print("Success! Model loaded with relative path")
    print("Model type:", type(model))
    print("Landmarks:", landmarks)
except Exception as e:
    print("Failed to load with relative path:", str(e))

# Test loading model with absolute path
print("\nTrying to load with absolute path...")
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Script directory:", script_dir)
    
    model_path = os.path.join(script_dir, 'safety_model.pkl')
    landmarks_path = os.path.join(script_dir, 'landmarks.pkl')
    
    print("Model path:", model_path)
    print("Landmarks path:", landmarks_path)
    
    model = joblib.load(model_path)
    landmarks = joblib.load(landmarks_path)
    print("Success! Model loaded with absolute path")
    print("Model type:", type(model))
    print("Landmarks:", landmarks)
except Exception as e:
    print("Failed to load with absolute path:", str(e))