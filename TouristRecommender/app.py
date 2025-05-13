import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load the dataset (update the path if needed)
df = pd.read_csv("TouristRecommender/Tourist.csv")

# Encode City and Category globally
le_city = LabelEncoder()
le_cat = LabelEncoder()
df['City_encoded'] = le_city.fit_transform(df['City'])
df['Category_encoded'] = le_cat.fit_transform(df['Category'])

# Streamlit UI
st.title("🏝️ Tourist Destination Recommender")
st.write("Select your preferences:")

city = st.selectbox("Choose City", df["City"].unique())
category = st.selectbox("Choose Category", df["Category"].unique())
rating_filter = st.selectbox("Filter by Rating", ["All", "Above 4", "Below 4"])

if st.button("Recommend"):
    # Filter by city and category
    filtered_df = df[(df["City"] == city) & (df["Category"] == category)]

    # Apply rating filter
    if rating_filter == "Above 4":
        filtered_df = filtered_df[filtered_df["Ratings_x"] >= 4.0]
    elif rating_filter == "Below 4":
        filtered_df = filtered_df[filtered_df["Ratings_x"] < 4.0]

    if filtered_df.empty:
        st.warning("No places match your criteria.")
    else:
        # Encode and scale
        filtered_df['City_encoded'] = le_city.transform(filtered_df['City'])
        filtered_df['Category_encoded'] = le_cat.transform(filtered_df['Category'])

        features = filtered_df[['City_encoded', 'Category_encoded']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Train KNN
        n_neighbors = min(6, len(filtered_df))
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(scaled_features)

        # Prepare input vector
        city_code = le_city.transform([city])[0]
        cat_code = le_cat.transform([category])[0]
        input_vec = scaler.transform([[city_code, cat_code]])

        # Get recommendations
        _, indices = knn.kneighbors(input_vec)

        st.subheader("Recommended Places:")

        for idx in indices[0]:
            place = filtered_df.iloc[idx]
            st.markdown(f"**{place['Place']}**  \n{place['Place_desc']}  \n⭐ {place['Ratings_x']}")
