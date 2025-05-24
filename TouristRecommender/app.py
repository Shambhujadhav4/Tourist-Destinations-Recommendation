import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df = pd.read_csv("Tourist.csv")
d = pd.read_csv("Tourist5.csv")

# Encoded City and Category
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
    filtered_df = df[(df["City"] == city) & (df["Category"] == category)].copy()

    # Apply rating filter
    if rating_filter == "Above 4":
        filtered_df = filtered_df[filtered_df["Ratings_x"] >= 4.0]
    elif rating_filter == "Below 4":
        filtered_df = filtered_df[filtered_df["Ratings_x"] < 4.0]

    if filtered_df.empty:
        st.warning("No places match your criteria.")
    else:
        
        #  Display all city images
        city_images = d[d["City"] == city]

        if not city_images.empty:
            st.subheader(f"📸 Images of {city}:")
    # Filter all columns starting with 'url'
            url_columns = [col for col in city_images.columns if col.lower().startswith("url")]
    
    # Collect valid image URLs
            image_urls = []
            for col in url_columns:
                urls = city_images[col].dropna().tolist()
                image_urls.extend([u for u in urls if isinstance(u, str) and u.startswith("http")])
    
    # Display images
            if image_urls:
                cols = st.columns(min(len(image_urls), 3))  # Max 4 images per row
                for i, url in enumerate(image_urls):
                    with cols[i % len(cols)]:
                        st.image(url, use_container_width=True)

        if 'City_desc' in df.columns:
            city_desc = df[df['City'] == city]['City_desc'].dropna().unique()
            if len(city_desc) > 0:
                st.info(f"**About {city}:** {city_desc[0]}")

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

        # Loop inside the button block
        for idx in indices[0]:
            place = filtered_df.iloc[idx]
            st.write(f"### 🏞️ {place['Place']}")
            st.write(f"**Description:** {place['Place_desc']}")
            st.write(f"**Rating:** ⭐ {place['Ratings_x']}")
            st.write(f"**Best Time to Visit:** {place['Best_time_to_visit']}")
            st.markdown("---")
