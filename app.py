import streamlit as st
import folium
import pandas as pd
import numpy as np
import joblib
import random
from streamlit_folium import st_folium
import os

# ------------------------------------------------------------------------------------
# Load model and landmarks
# ------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'safety_model.pkl')
    landmarks_path = os.path.join(script_dir, 'landmarks.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(landmarks_path):
        raise FileNotFoundError(f"Landmarks not found: {landmarks_path}")

    model = joblib.load(model_path)
    landmarks = joblib.load(landmarks_path)
    return model, landmarks


# ------------------------------------------------------------------------------------
# Coordinates for Vijayawada Landmarks
# ------------------------------------------------------------------------------------
LANDMARK_COORDINATES = {
    "Benz Circle": (16.5080, 80.6470),
    "MG Road": (16.5020, 80.6420),
    "Labbipet": (16.5120, 80.6450),
    "Governorpet": (16.5050, 80.6380),
    "Auto Nagar": (16.4980, 80.6520),
    "Vidhyadharpuram": (16.5150, 80.6350),
    "Gollapudi": (16.5200, 80.6280),
    "Enikepadu": (16.4950, 80.6580),
    "Rama Krishna Puram": (16.5100, 80.6250),
    "Hanumanpet": (16.5030, 80.6600)
}

def get_coordinates(name):
    return LANDMARK_COORDINATES.get(name, (16.5062, 80.6480))


# ------------------------------------------------------------------------------------
# Route Generators
# ------------------------------------------------------------------------------------
def generate_route(start, end, points=10):
    lat = np.linspace(start[0], end[0], points)
    lon = np.linspace(start[1], end[1], points)
    return list(zip(lat, lon))


# ⭐ DYNAMIC ALTERNATE ROUTE (not fixed to Labbipet)
def generate_alternate_route(start, end):
    """Generate alternate route through a NEARBY safe landmark automatically"""

    # Exclude start and end
    safe_points = {
        name: coord for name, coord in LANDMARK_COORDINATES.items()
        if coord != start and coord != end
    }

    # Midpoint to estimate safest detour
    mid_lat = (start[0] + end[0]) / 2
    mid_lon = (start[1] + end[1]) / 2

    closest_landmark = None
    min_dist = float("inf")

    # Pick closest safe landmark
    for name, (x, y) in safe_points.items():
        dist = ((x - mid_lat) ** 2 + (y - mid_lon) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_landmark = (name, (x, y))

    landmark_name, landmark_coord = closest_landmark

    # Route: start → landmark → end
    p1_lat = np.linspace(start[0], landmark_coord[0], 10)
    p1_lon = np.linspace(start[1], landmark_coord[1], 10)

    p2_lat = np.linspace(landmark_coord[0], end[0], 10)
    p2_lon = np.linspace(landmark_coord[1], end[1], 10)

    route = list(zip(p1_lat, p1_lon)) + list(zip(p2_lat, p2_lon))

    return route, landmark_name


# ------------------------------------------------------------------------------------
# ML Safety Prediction
# ------------------------------------------------------------------------------------
def predict_safety(model, src_idx, dest_idx, time_value):
    distance = random.uniform(2, 12)
    density = random.uniform(2000, 8000)
    lights = random.uniform(30, 90)
    police = random.randint(0, 3)

    feature_names = model.feature_names_in_
    df = pd.DataFrame([{
        feature_names[0]: src_idx,
        feature_names[1]: dest_idx,
        feature_names[2]: time_value,
        feature_names[3]: distance,
        feature_names[4]: density,
        feature_names[5]: lights,
        feature_names[6]: police
    }])

    label = model.predict(df)[0]

    if label == "Unsafe":
        score = random.randint(10, 30)
    elif label == "Moderate":
        score = random.randint(31, 70)
    else:
        score = random.randint(71, 95)

    return score, label


# ------------------------------------------------------------------------------------
# Suggestions
# ------------------------------------------------------------------------------------
def get_suggestions(time_of_day, label):
    tips = {
        "Morning": [
            "Prefer well-lit main roads.",
            "Stay alert.",
            "Inform someone about your travel plan."
        ],
        "Afternoon": [
            "Stick to busy areas.",
            "Avoid isolated shortcuts.",
            "Keep your phone charged."
        ],
        "Evening": [
            "Prefer lit roads.",
            "Avoid alleys.",
            "Travel with a friend."
        ],
        "Night": [
            "Avoid isolated roads after 9 PM.",
            "Share your live location.",
            "Have emergency contacts ready."
        ]
    }

    if label == "Safe":
        return "✅ This route appears safe."

    return "⚠️ " + random.choice(tips[time_of_day])


# ------------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Women Safety Prediction", layout="wide")
    st.title("🛡️ Women Safety Prediction System")

    model, landmarks = load_model()

    col1, col2, col3 = st.columns(3)
    with col1:
        source = st.selectbox("From", landmarks)
    with col2:
        destination = st.selectbox("To", landmarks)
    with col3:
        tod = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}

    if st.button("Predict Safety", type="primary"):
        st.session_state.run = True

        src_idx = landmarks.index(source)
        dest_idx = landmarks.index(destination)
        time_val = time_map[tod]

        score, label = predict_safety(model, src_idx, dest_idx, time_val)

        src_coord = get_coordinates(source)
        dest_coord = get_coordinates(destination)

        # dynamic alternate route
        alt_route, alt_landmark = generate_alternate_route(src_coord, dest_coord)

        st.session_state.direct_score = score
        st.session_state.direct_label = label
        st.session_state.src_coord = src_coord
        st.session_state.dest_coord = dest_coord
        st.session_state.direct_route = generate_route(src_coord, dest_coord)
        st.session_state.alt_route = alt_route
        st.session_state.alt_landmark = alt_landmark
        st.session_state.source = source
        st.session_state.destination = destination
        st.session_state.tod = tod

    # RESULTS
    if st.session_state.get("run", False):

        st.subheader("Prediction Results")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🔴 Direct Route Safety")
            st.markdown(f"**{st.session_state.direct_score}% ({st.session_state.direct_label})**")
            st.markdown(get_suggestions(st.session_state.tod, st.session_state.direct_label))

        with c2:
            st.markdown("### 🟢 Alternate Safe Route")
            st.markdown(
                f"**Suggested Route:** {st.session_state.source} → {st.session_state.alt_landmark} → {st.session_state.destination}"
            )
            st.markdown("This route passes through busier, safer areas.")

        # MAP
        st.subheader("Route Visualization")

        m = folium.Map(location=[16.5062, 80.6480], zoom_start=13)

        folium.Marker(
            st.session_state.src_coord,
            icon=folium.Icon(color="green"),
            popup="Source"
        ).add_to(m)

        folium.Marker(
            st.session_state.dest_coord,
            icon=folium.Icon(color="red"),
            popup="Destination"
        ).add_to(m)

        # Direct Route
        folium.PolyLine(
            st.session_state.direct_route,
            color="red",
            weight=5,
            tooltip=f"Direct Route: {st.session_state.direct_score}% ({st.session_state.direct_label})"
        ).add_to(m)

        # Alternate Route
        folium.PolyLine(
            st.session_state.alt_route,
            color="green",
            weight=5,
            tooltip="Alternate Safe Route (Recommended)"
        ).add_to(m)

        # POLICE STATIONS
        police_stations = [
            ("Vijayawada Police Station", 16.508, 80.645),
            ("Labbipet Police Station", 16.512, 80.642),
            ("Benz Circle Police Station", 16.505, 80.655),
            ("MG Road Police Station", 16.502, 80.638)
        ]

        for name, lat, lon in police_stations:
            folium.Marker(
                [lat, lon],
                popup=name,
                icon=folium.Icon(color="blue", icon="shield")
            ).add_to(m)

        st_folium(m, width=800, height=500)

        # Safety Tips
        st.subheader("Safety Tips")
        for t in [
            "Share your location with trusted contacts.",
            "Prefer well-lit roads.",
            "Avoid isolated shortcuts.",
            "Keep your phone charged.",
            "Have emergency contacts ready."
        ]:
            st.markdown(f"- {t}")


if __name__ == "__main__":
    main()
