# app_ui.py
import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import quote
import time

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Smart Tourist Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Minimal CSS for modern look
# ---------------------------
st.markdown(
    """
    <style>
    /* --- Optimized CSS - Reduced redundancy --- */
    :root {
        --primary-gradient: linear-gradient(135deg, rgba(102,126,234,0.95), rgba(118,75,162,0.95));
        --light-bg: linear-gradient(180deg, #f6f9ff 0%, #ffffff 40%);
        --dark-bg: linear-gradient(180deg, #0f172a 0%, #071033 60%);
        --card-light: linear-gradient(180deg, #ffffff, #fbfdff);
        --card-dark: #0b1220;
        --text-light: #111827;
        --text-dark: #e6eef8;
        --text-muted-light: #6b7280;
        --text-muted-dark: #94a3b8;
    }
    
    .stApp {
        background: var(--light-bg) !important;
        font-family: "Inter", "Segoe UI", Tahoma, Geneva, Verdana, sans-serif !important;
        color: var(--text-light) !important;
    }
    
    .hero {
        border-radius: 14px !important;
        padding: 28px !important;
        margin-bottom: 18px !important;
        background: var(--primary-gradient) !important;
        color: white !important;
        box-shadow: 0 10px 30px rgba(99,102,241,0.12) !important;
    }
    
    .hero h1, .hero p { 
        margin: 0 !important; 
        color: white !important;
    }
    
    .hero h1 { font-size: 2.2rem !important; font-weight: 700 !important; }
    .hero p { margin-top: 6px !important; opacity: 0.95 !important; }

    .place-card {
        background: var(--card-light) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 6px 18px rgba(16,24,40,0.06) !important;
        transition: transform 0.18s ease !important;
        border: 1px solid rgba(16,24,40,0.04) !important;
        margin-bottom: 20px !important;
        color: var(--text-light) !important;
    }
    
    .place-card:hover { transform: translateY(-6px) !important; }
    .place-card .img { width: 100% !important; height: 180px !important; object-fit: cover !important; }
    .place-card .body { padding: 14px !important; }
    .place-card h3 { margin: 0 0 6px 0 !important; font-size: 1.05rem !important; color: var(--text-light) !important; }
    .place-card p { margin: 0 !important; color: var(--text-muted-light) !important; font-size: 0.92rem !important; line-height: 1.4 !important; }

    .btn {
        display: inline-block !important;
        padding: 8px 12px !important;
        border-radius: 10px !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        margin-right: 8px !important;
        border: none !important;
        transition: transform 0.2s ease !important;
    }
    
    .btn-primary { 
        background: linear-gradient(90deg,#667eea,#764ba2) !important; 
        color: white !important; 
    }
    
    .btn-primary:hover { transform: translateY(-2px) !important; }

    .muted { color: var(--text-muted-light) !important; font-size: 0.9rem !important; }
    .rating { font-weight: 700 !important; color: #b45309 !important; }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease !important;
    }
    
    .stButton > button:hover { transform: translateY(-2px) !important; }
    
    /* Dark theme - only when needed */
    @media (prefers-color-scheme: dark) {
        .stApp { background: var(--dark-bg) !important; color: var(--text-dark) !important; }
        .place-card { background: var(--card-dark) !important; color: var(--text-dark) !important; border-color: rgba(255,255,255,0.04) !important; }
        .place-card h3 { color: var(--text-dark) !important; }
        .place-card p { color: #cbd5e1 !important; }
        .muted { color: var(--text-muted-dark) !important; }
        .rating { color: #fbbf24 !important; }
    }
    
    /* Auto-scroll to top functionality */
    html {
        scroll-behavior: smooth;
    }
    
    /* Ensure the top anchor is visible */
    #top {
        position: absolute;
        top: 0;
        left: 0;
        width: 1px;
        height: 1px;
        visibility: hidden;
    }
    </style>
    
    <script>
    // Simple scroll to top function
    function scrollToTop() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    // Scroll to top when page loads
    window.addEventListener('load', scrollToTop);
    
    // Scroll to top when any form element changes
    document.addEventListener('change', function(e) {
        if (e.target.matches('select, input, textarea')) {
            setTimeout(scrollToTop, 50);
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
def get_google_maps_url(place_name: str, city: str) -> str:
    query = f"{place_name}, {city}"
    return f"https://www.google.com/maps/search/?api=1&query={quote(query)}"

def placeholder_images(n=3):
    images = [
        "https://picsum.photos/800/600?random=1",
        "https://picsum.photos/800/600?random=2", 
        "https://picsum.photos/800/600?random=3",
        "https://picsum.photos/800/600?random=4",
        "https://picsum.photos/800/600?random=5",
    ]
    return images[:n]

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    # try loading dataset from expected path
    try:
        df = pd.read_csv("TouristRecommender/Touristnew.csv")
        d = pd.read_csv("TouristRecommender/Tourist5.csv")
    except Exception as e:
        # If failing, return None to let UI show fallback message
        return None, None

    # Add safe defaults if missing - optimized
    defaults = {
        'Budget_Range': ['Low (<₹5000)', 'Medium (₹5000-15000)', 'High (>₹15000)'],
        'Duration': ['1-2 days', '3-5 days', '1 week+'],
        'Travel_Type': ['Solo','Couple','Family','Group']
    }
    
    for col, choices in defaults.items():
        if col not in df.columns:
            df[col] = np.random.choice(choices, len(df))
    
    if 'Place_desc' not in df.columns:
        df['Place_desc'] = df.get('Place', '').astype(str) + " — A wonderful place to visit."
    if 'City_desc' not in df.columns:
        df['City_desc'] = ""
        
    # Ensure Ratings_x exists (fallback to Ratings if present)
    if 'Ratings_x' not in df.columns:
        if 'Ratings' in df.columns:
            df['Ratings_x'] = df['Ratings']
        else:
            df['Ratings_x'] = np.clip(np.round(np.random.uniform(3.0, 4.9, size=len(df)), 1), 1.0, 5.0)
    return df, d

# ---------------------------
# Load data
# ---------------------------
df, d = load_data()
if df is None or d is None:
    st.warning("Could not find dataset files at `Touristnew.csv` and/or `Tourist5.csv`.\n\n"
               "Make sure the CSV files are in the same directory as app_ui.py. Using demo placeholders for now.")
    # Create a tiny demo frame to allow UI to operate
    demo = {
        "City": ["Goa", "Goa", "Rishikesh", "Manali", "Agra", "Jaipur"],
        "Category": ["Beaches","Nightlife","Adventure","Hills","Heritage","Heritage"],
        "Place": ["Calangute Beach","Baga Night Market","Ganga River","Solang Valley","Taj Mahal","Amber Fort"],
        "Place_desc": [
            "Sandy beach, popular with tourists.",
            "Lively night market full of food & music.",
            "Sacred river, rafting & scenic ghats.",
            "Snowy peaks and adventure sports.",
            "World-famous monument of love.",
            "Historic fort with panoramic views."
        ],
        "Ratings_x": [4.2, 4.0, 4.5, 4.3, 4.9, 4.4],
        "Distance": ["2 km","3 km","1 km","5 km","0.5 km","6 km"],
        "Best_time_to_visit": ["Nov-Feb","Oct-Jan","Sep-May","Dec-Feb","Oct-Mar","Nov-Feb"],
        "City_desc": ["Goa is a beach lover's paradise."]*6
    }
    df = pd.DataFrame(demo)
    d = pd.DataFrame({"City": ["Goa", "Rishikesh", "Manali", "Agra", "Jaipur"],
                      "Image_URL": placeholder_images(5)})

# ---------------------------
# Session state: feedback and user name
# ---------------------------
if "liked_places" not in st.session_state:
    st.session_state.liked_places = set()
if "disliked_places" not in st.session_state:
    st.session_state.disliked_places = set()
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# ---------------------------
# Name Input (if not set)
# ---------------------------
if st.session_state.user_name is None:
    st.markdown("---")
    st.markdown("### 👋 Welcome to Tourist Destination Recommender!")
    st.markdown("Please enter your name to get started:")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_name = st.text_input("Your Name", placeholder="Enter your name here", key="name_input")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Continue", type="primary", use_container_width=True, key="continue_btn"):
                if user_name and user_name.strip():
                    st.session_state.user_name = user_name.strip()
                    st.success(f"Welcome, {user_name.strip()}!")
                    st.rerun()
                else:
                    st.error("Please enter a valid name!")
        
        with col_b:
            if st.button("Skip", use_container_width=True, key="skip_btn"):
                st.session_state.user_name = "Guest"
                st.rerun()
    
    st.markdown("---")
    st.stop()  # Stop execution until name is entered

# ---------------------------
# Sidebar (filters, profile)
# ---------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/747/747376.png", width=72)
    st.markdown(f"### 👋 Hello, {st.session_state.user_name}")
    st.markdown("Plan your next trip — fast and pretty ✨")
    st.divider()

    # Theme toggle (enhanced with system detection)
    st.markdown("#### 🎨 Theme Settings")
    theme = st.radio("Theme Preference", options=["🌞 Light", "🌙 Dark", "🔄 Auto (System)"], index=2)
    
    if theme == "🌙 Dark":
        st.markdown(
            """<style> 
                .stApp { 
                    background: linear-gradient(180deg,#0f172a 0%, #071033 60%) !important; 
                    color: #e6eef8 !important; 
                } 
                .place-card { 
                    background: #0b1220 !important; 
                    color: #e6eef8 !important; 
                    border: 1px solid rgba(255,255,255,0.04) !important; 
                }
                .place-card h3 { color: #e6eef8 !important; }
                .place-card p { color: #cbd5e1 !important; }
                .hero { color: white !important; }
                .hero h1 { color: white !important; }
                .hero p { color: white !important; }
                .muted { color: #94a3b8 !important; }
                .rating { color: #fbbf24 !important; }
                .css-1d391kg {
                    background: rgba(11, 18, 32, 0.9) !important;
                    border: 1px solid rgba(255,255,255,0.1) !important;
                }
            </style>""",
            unsafe_allow_html=True,
        )
    elif theme == "🌞 Light":
        st.markdown(
            """<style> 
                .stApp { 
                    background: linear-gradient(180deg, #f6f9ff 0%, #ffffff 40%) !important; 
                    color: #111827 !important; 
                } 
                .place-card { 
                    background: linear-gradient(180deg, #ffffff, #fbfdff) !important; 
                    color: #111827 !important; 
                    border: 1px solid rgba(16,24,40,0.04) !important; 
                }
                .place-card h3 { color: #111827 !important; }
                .place-card p { color: #4b5563 !important; }
                .hero { color: white !important; }
                .hero h1 { color: white !important; }
                .hero p { color: white !important; }
                .muted { color: #6b7280 !important; }
                .rating { color: #b45309 !important; }
                .css-1d391kg {
                    background: rgba(255, 255, 255, 0.9) !important;
                    border: 1px solid rgba(16,24,40,0.1) !important;
                }
            </style>""",
            unsafe_allow_html=True,
        )
    # Auto theme uses the CSS media queries defined above

    st.divider()
    # Quick search & filters
    st.markdown("#### 🔎 Quick search")
    search_text = st.text_input("Search place / city / category", value="", placeholder="e.g. Goa, Beaches")

    st.markdown("#### 🗂 Filters")
    # City and category dropdowns
    cities = sorted(df["City"].dropna().unique().tolist())
    categories = sorted(df["Category"].dropna().unique().tolist())
    city_filter = st.selectbox("City", options=["Any"] + cities, index=0)
    category_filter = st.selectbox("Category", options=["Any"] + categories, index=0)

    st.markdown("Budget & Duration")
    budget_filter = st.selectbox("Budget", options=["Any","Low (<₹5000)","Medium (₹5000-15000)","High (>₹15000)"], index=0)
    duration_filter = st.selectbox("Duration", options=["Any","1-2 days","3-5 days","1 week+"], index=0)
    travel_type_filter = st.selectbox("Travel Type", options=["Any","Solo","Couple","Family","Group"], index=0)

    st.divider()

    # Small legend / hints
    st.markdown("**Tip:** Click a place card's Like / Dislike to build your preferences. Download current recommendations when ready.")
    
    # Logout option
    if st.button("🔄 Change Name", help="Reset and enter a new name"):
        st.session_state.user_name = None
        st.rerun()
    
    st.divider()

# ---------------------------
# Scroll anchor and Top hero
# ---------------------------
# Create a scroll anchor at the top
scroll_anchor = st.empty()
scroll_anchor.markdown('<div id="top"></div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="hero">
        <h1>🧭 Smart Tourist Destination Recommender</h1>
        <p>Discover curated places by city & category. Visual cards, instant feedback (like/dislike), and downloadable lists.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Filtering logic - Optimized
# ---------------------------
# Start with whole df
results = df.copy()

# Apply search text - optimized with vectorized operations
q = search_text.strip().lower()
if q:
    # Use vectorized string operations for better performance
    mask = (
        results['Place'].str.lower().str.contains(q, na=False) |
        results['City'].str.lower().str.contains(q, na=False) |
        results['Category'].str.lower().str.contains(q, na=False)
    )
    results = results[mask]

# Apply dropdown filters - optimized
filters = {
    'City': city_filter,
    'Category': category_filter,
    'Budget_Range': budget_filter,
    'Duration': duration_filter,
    'Travel_Type': travel_type_filter
}

for col, value in filters.items():
    if value != "Any":
        results = results[results[col] == value]

# Sort by rating (descending) - only if column exists
if 'Ratings_x' in results.columns:
    results = results.sort_values(by='Ratings_x', ascending=False)

# ---------------------------
# Summary row
# ---------------------------
col1, col2, col3 = st.columns([1,2,1])
with col1:
    st.metric("🧭 Results", len(results))
with col2:
    # natural-language summary
    top_city = city_filter if city_filter != "Any" else (results['City'].mode()[0] if not results.empty else "—")
    top_category = category_filter if category_filter != "Any" else (results['Category'].mode()[0] if not results.empty else "—")
    st.markdown(f"**Summary:** Found **{len(results)}** places — top suggestions for **{top_category}** in **{top_city}**.")
with col3:
    st.metric("❤️ Liked", len(st.session_state.liked_places))

st.markdown("")

# ---------------------------
# Display recommendations as responsive grid
# ---------------------------
def get_place_image(place_name, city_name):
    # Use ONLY Picsum Photos with random images
    try:
        # Generate consistent random number based on place name
        random_num = hash(place_name) % 1000
        return f"https://picsum.photos/800/600?random={random_num}"
    except Exception:
        pass
    
    # Fallback: placeholder image
    return placeholder_images(1)[0]

# Number of columns for cards (responsive-ish)
cols_num = 3
cards = st.container()

# Limit results for better performance
max_results = 12  # Show max 12 results to prevent lag
display_results = results.head(max_results)

with cards:
    if len(display_results) > 0:
        rows = int(np.ceil(len(display_results) / cols_num))
        idx = 0
        for r in range(rows):
            cols = st.columns(cols_num, gap="large")
            for c in cols:
                if idx >= len(display_results):
                    c.empty()
                    idx += 1
                    continue
                    
                row = display_results.iloc[idx]
                place_name = row.get("Place", "Unknown Place")
                city_name = row.get("City", "Unknown City")
                desc = str(row.get("Place_desc", ""))[:200]  # Reduced from 240
                rating = row.get("Ratings_x", None)
                distance = row.get("Distance", "—")
                best_time = row.get("Best_time_to_visit", "—")
                budget = row.get("Budget_Range", "—")
                travel_type = row.get("Travel_Type", "—")

                # image
                img_url = get_place_image(place_name, city_name)

                # Simplified card markup for better performance with error handling
                card_html = f"""
                <div class="place-card">
                    <img src="{img_url}" class="img" alt="{place_name}" loading="lazy" onerror="this.src='https://via.placeholder.com/800x600/4f46e5/ffffff?text={place_name.replace(' ', '+')}'">
                    <div class="body">
                        <h3>🏝️ {place_name} <span style="float:right;" class="rating">{"⭐ " + str(rating) if pd.notna(rating) else ""}</span></h3>
                        <div class="muted">{city_name} · {distance} · Best: {best_time}</div>
                        <p style="margin-top:8px;">{desc}{"..." if len(desc) >= 200 else ""}</p>
                    </div>
                </div>
                """

                c.markdown(card_html, unsafe_allow_html=True)

                # Simplified buttons for better performance
                like_key = f"like__{idx}__{place_name}"
                dislike_key = f"dislike__{idx}__{place_name}"
                maps_url = get_google_maps_url(place_name, city_name)

                with c:
                    col_a, col_b, col_c = st.columns([1,1,2])
                    with col_a:
                        if st.button(("❤️" if place_name in st.session_state.liked_places else "👍"),
                                     key=like_key):
                            st.session_state.liked_places.add(place_name)
                            st.session_state.disliked_places.discard(place_name)
                            st.rerun()
                    with col_b:
                        if st.button(("💔" if place_name in st.session_state.disliked_places else "👎"),
                                     key=dislike_key):
                            st.session_state.disliked_places.add(place_name)
                            st.session_state.liked_places.discard(place_name)
                            st.rerun()
                    with col_c:
                        st.markdown(f'<a class="btn btn-primary" href="{maps_url}" target="_blank">🗺️ Maps</a>', unsafe_allow_html=True)
                idx += 1
    else:
        st.info("No results match your search/filters. Try changing filters or clearing the search box.")

st.markdown("---")

# ---------------------------
# Preferences / analytics (compact)
# ---------------------------
st.subheader("📊 Your Preferences & Quick Insights")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total results", len(results))
with col2:
    st.metric("Liked", len(st.session_state.liked_places))
with col3:
    st.metric("Disliked", len(st.session_state.disliked_places))

# Show detailed feedback summary
if st.session_state.liked_places or st.session_state.disliked_places:
    st.markdown("**Your Feedback Summary:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👍 Liked Places:**")
        if st.session_state.liked_places:
            for place in sorted(st.session_state.liked_places):
                st.markdown(f"• {place}")
        else:
            st.markdown("*No places liked yet*")
    
    with col2:
        st.markdown("**👎 Disliked Places:**")
        if st.session_state.disliked_places:
            for place in sorted(st.session_state.disliked_places):
                st.markdown(f"• {place}")
        else:
            st.markdown("*No places disliked yet*")
else:
    st.info("Like or dislike some places to see your feedback summary here!")

st.markdown("---")

# ---------------------------
# Download current recommendations CSV
# ---------------------------
if not results.empty:
    csv_bytes = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download current recommendations (CSV)",
        data=csv_bytes,
        file_name="recommendations.csv",
        mime="text/csv"
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <div style="padding:18px; border-radius:8px; margin-top:20px; text-align:center; color:#6b7280;">
        Built with ❤️ by <strong>Shambhuraje Jadhav</strong> — Tourist Destination Recommender • 2025
    </div>
    """,
    unsafe_allow_html=True
)
