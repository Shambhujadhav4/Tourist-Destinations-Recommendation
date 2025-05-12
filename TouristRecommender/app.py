import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df = pd.read_csv("TouristRecommender/Tourist.csv")

# Encode City and Category
le_city = LabelEncoder()
le_cat = LabelEncoder()
df['City_encoded'] = le_city.fit_transform(df['City'])
df['Category_encoded'] = le_cat.fit_transform(df['Category'])

# Feature Selection
features = df[['City_encoded', 'Category_encoded', 'Ratings_x']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# KNN Model
knn = NearestNeighbors(n_neighbors=6)
knn.fit(scaled_features)

# Streamlit UI
st.title("🏝️ Tourist Destination Recommender")
st.write("Select your preferences:")

city = st.selectbox("Choose City", df["City"].unique())
category = st.selectbox("Choose Category", df["Category"].unique())

if st.button("Recommend"):
    city_code = le_city.transform([city])[0]
    cat_code = le_cat.transform([category])[0]
    avg_rating = df["Ratings_x"].mean()
    input_vec = scaler.transform([[city_code, cat_code, avg_rating]])
    
    _, indices = knn.kneighbors(input_vec)
    
    st.subheader("Recommended Places:")
    for idx in indices[0][1:]:  # Skip the first (it will match itself)
        place = df.iloc[idx]
        st.markdown(f"**{place['Place']}**  \n{place['Place_desc']}  \n⭐ {place['Ratings_x']}")

