import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Page config and styling
st.set_page_config(page_title="Tourist Recommender", page_icon="🏝️", layout="wide")
st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stApp {max-width: 1200px; margin: auto;}
    .card {
        background-color: #1c2b3a;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        border-left: 5px solid #00bfff;
        transition: transform 0.3s;
    }
    .card:hover {
        transform: scale(1.01);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .rating {
        color: #ffd700;
        font-weight: bold;
        font-size: 1.2rem;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    p {
        color: #ffffff !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    .image-card img {
        border-radius: 10px;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.3);
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("TouristRecommender/Touristnew.csv")
    d = pd.read_csv("TouristRecommender/Tourist5.csv")
    return df, d

def encode_features(df):
    le_city = LabelEncoder()
    le_cat = LabelEncoder()
    df['City_encoded'] = le_city.fit_transform(df['City'])
    df['Category_encoded'] = le_cat.fit_transform(df['Category'])
    return df, le_city, le_cat

def load_city_images(city, d):
    city_images = d[d["City"] == city]
    urls = [url for col in city_images.columns if col.lower().startswith("url") for url in city_images[col].dropna()]
    return urls

def display_image_gallery(urls):
    if urls:
        cols = st.columns(min(len(urls), 3))
        for i, url in enumerate(urls):
            with cols[i % len(cols)]:
                st.image(url, use_container_width=True)

def knn_recommendations(df, le_city, le_cat, city, category):
    df['City_encoded'] = le_city.transform(df['City'])
    df['Category_encoded'] = le_cat.transform(df['Category'])
    features = df[['City_encoded', 'Category_encoded']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = NearestNeighbors(n_neighbors=min(6, len(df)))
    model.fit(scaled)
    vec = scaler.transform([[le_city.transform([city])[0], le_cat.transform([category])[0]]])
    _, indices = model.kneighbors(vec)
    return df.iloc[indices[0]]

# Load data
df, d = load_data()
df, le_city, le_cat = encode_features(df)

# UI - Sidebar
st.title("🧭 Smart Tourist Destination Recommender")
st.markdown("Get smart recommendations based on your interest 🔍")

with st.sidebar:
    st.header("🔧 Filter Preferences")
    city = st.selectbox("🌍 Choose City", df["City"].unique())
    category = st.selectbox("🗂️ Choose Category", df["Category"].unique())
    recommend = st.button("✨ Recommend Now")

if recommend:
    filtered_df = df[(df["City"] == city) & (df["Category"] == category)]

    if filtered_df.empty:
        st.warning("⚠️ No places found with the selected filters.")
    else:
        st.subheader(f"📸 Highlights of {city}")
        image_urls = load_city_images(city, d)

        with st.spinner("🔄 Loading images..."):
            display_image_gallery(image_urls)

        city_desc = df[df['City'] == city]['City_desc'].dropna().unique()
        if city_desc.size:
            st.info(f"**🗽 About {city}:** {city_desc[0]}")

        with st.spinner("🔄 Generating smart recommendations..."):
            recommendations = knn_recommendations(filtered_df, le_city, le_cat, city, category)
            recommendations = recommendations.sort_values(by='Ratings_x', ascending=False)

            st.subheader("🏆 Top Recommended Places")
            cols = st.columns(2)
            for i, (_, place) in enumerate(recommendations.iterrows()):
                stars = "⭐" * int(round(place['Ratings_x']))
                with cols[i % 2]:
                    st.markdown(f"""<div class=\"card\">
                        <h3>{i+1}. 🏜️ {place['Place']}</h3>
                        <p><strong>📖 Description:</strong> {place['Place_desc']}</p>
                        <p><strong>🕒 Best Time to Visit:</strong> {place['Best_time_to_visit']}</p>
                        <p class=\"rating\">{stars} ({place['Ratings_x']})</p>
                    </div>""", unsafe_allow_html=True)

            st.download_button("📅 Download as CSV", recommendations.to_csv(index=False), "recommendations.csv", "text/csv")
