# app_ui.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from urllib.parse import quote
from html import escape
import time
import os
from pathlib import Path
import base64

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Smart Tourist Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Smart Tourist Destination Recommender - Find your perfect travel destination!"
    }
)

# ---------------------------
# TripAdvisor-style CSS
# ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=Manrope:wght@400;500;700&display=swap');

    /* --- Horizon Atlas: Travel-magazine Theme --- */
    :root {
        --brand-teal: #2a9d8f;
        --brand-coral: #e76f51;
        --brand-gold: #e9c46a;
        --brand-ink: #2f2a22;
        --bg-light: #f7f3ea;
        --bg-card: #fffdf8;
        --bg-soft: #efe7d7;
        --text-primary: #2f2a22;
        --text-secondary: #61584b;
        --text-muted: #7a705e;
        --border-color: #d8ccb6;
        --card-shadow: 0 8px 24px rgba(87, 63, 24, 0.08);
        --card-shadow-hover: 0 14px 30px rgba(42, 157, 143, 0.18);
    }
    
    .stApp {
        background:
            radial-gradient(circle at 10% 5%, rgba(233, 196, 106, 0.23), transparent 34%),
            radial-gradient(circle at 85% 3%, rgba(42, 157, 143, 0.16), transparent 28%),
            linear-gradient(180deg, #fffaf2 0%, #f7f3ea 42%, #f2ecdf 100%) !important;
        font-family: "Manrope", "Segoe UI", Arial, sans-serif !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        background: rgba(255, 253, 248, 0.78) !important;
        border: 1px solid rgba(216, 204, 182, 0.75) !important;
        border-radius: 18px !important;
        box-shadow: 0 12px 30px rgba(118, 95, 62, 0.08) !important;
    }
    
    /* Header/Navigation Bar */
    .header-nav {
        background: var(--bg-card) !important;
        padding: 14px 0 !important;
        margin-bottom: 24px !important;
        border-bottom: 1px solid var(--border-color) !important;
        box-shadow: 0 6px 20px rgba(87, 63, 24, 0.08) !important;
        border-radius: 12px !important;
    }
    
    .header-nav h1 {
        color: var(--brand-ink) !important;
        font-family: "Fraunces", Georgia, serif !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        display: inline-block !important;
    }
    
    /* Search Bar Hero */
    .search-hero {
        background: var(--bg-card) !important;
        border-radius: 8px !important;
        padding: 32px !important;
        margin-bottom: 32px !important;
        box-shadow: var(--card-shadow) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .search-hero h2 {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin: 0 0 16px 0 !important;
    }
    
    .search-hero p {
        color: var(--text-secondary) !important;
        margin: 0 !important;
        font-size: 0.95rem !important;
    }

    /* TripAdvisor-style Cards */
    .place-card {
        background: var(--bg-card) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: var(--card-shadow) !important;
        transition: all 0.3s ease !important;
        border: 1px solid var(--border-color) !important;
        margin-bottom: 24px !important;
        color: var(--text-primary) !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    .place-card:hover { 
        transform: translateY(-4px) !important;
        box-shadow: var(--card-shadow-hover) !important;
        border-color: var(--brand-teal) !important;
    }
    
    .place-card .img { 
        width: 100% !important; 
        height: 200px !important; 
        object-fit: cover !important;
        display: block !important;
        filter: saturate(1.04) contrast(1.03) !important;
    }

    .image-wrap {
        position: relative !important;
    }

    .image-overlay {
        position: absolute !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        padding: 10px 10px 12px 10px !important;
        background: linear-gradient(to top, rgba(31, 23, 14, 0.62), rgba(31, 23, 14, 0.0)) !important;
        display: flex !important;
        justify-content: flex-start !important;
    }

    .category-pill {
        background: rgba(255, 250, 242, 0.9) !important;
        color: #3e372c !important;
        border: 1px solid rgba(216, 204, 182, 0.9) !important;
        border-radius: 999px !important;
        padding: 4px 10px !important;
        font-size: 0.74rem !important;
        font-weight: 700 !important;
    }
    
    .place-card .body { 
        padding: 16px !important;
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    .place-card h3 { 
        margin: 0 0 8px 0 !important; 
        font-size: 1.15rem !important; 
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        line-height: 1.3 !important;
    }

    .place-title-link {
        color: var(--text-primary) !important;
        text-decoration: none !important;
        border-bottom: 1px dashed transparent !important;
        transition: all 0.2s ease !important;
    }

    .place-title-link:hover {
        color: var(--brand-coral) !important;
        border-bottom-color: var(--brand-coral) !important;
    }
    
    .place-card .location-info {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
        margin-bottom: 8px !important;
    }
    
    .place-card p { 
        margin: 0 !important; 
        color: var(--text-secondary) !important; 
        font-size: 0.9rem !important; 
        line-height: 1.5 !important;
        flex-grow: 1 !important;
    }
    
    /* Rating Badge - TripAdvisor Style */
    .rating-badge {
        display: inline-flex !important;
        align-items: center !important;
        background: rgba(233, 196, 106, 0.25) !important;
        color: #8f5b00 !important;
        padding: 4px 8px !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        margin-left: 8px !important;
        border: 1px solid rgba(233, 196, 106, 0.45) !important;
    }
    
    .rating-badge .rating-value {
        color: #8f5b00 !important;
        font-weight: 700 !important;
    }

    .btn {
        display: inline-block !important;
        padding: 10px 16px !important;
        border-radius: 6px !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
        margin-right: 8px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .btn-primary { 
        background: linear-gradient(120deg, var(--brand-teal), #1f8d80) !important; 
        color: white !important; 
    }
    
    .btn-primary:hover { 
        background: linear-gradient(120deg, #1f8d80, #157569) !important;
        transform: translateY(-1px) !important;
    }
    
    .btn-outline {
        background: transparent !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    .btn-outline:hover {
        border-color: var(--brand-coral) !important;
        color: var(--brand-coral) !important;
    }

    .card-map-btn {
        display: block !important;
        width: 100% !important;
        text-align: center !important;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        font-size: 0.92rem !important;
        box-shadow: 0 8px 16px rgba(42, 157, 143, 0.18) !important;
    }

    .muted { 
        color: var(--text-muted) !important; 
        font-size: 0.875rem !important; 
    }
    
    .stButton > button {
        background: linear-gradient(120deg, var(--brand-coral), #d95d3f) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 6px 12px rgba(231, 111, 81, 0.22) !important;
    }
    
    .stButton > button:hover { 
        background: linear-gradient(120deg, #d95d3f, #c94f33) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Filter Section */
    .filter-section {
        background: var(--bg-card) !important;
        padding: 16px !important;
        border-radius: 8px !important;
        margin-bottom: 24px !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Results Summary */
    .results-summary {
        background: var(--bg-card) !important;
        padding: 16px 20px !important;
        border-radius: 8px !important;
        margin-bottom: 24px !important;
        border: 1px solid var(--border-color) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }

    /* Magazine-style accents */
    .header-nav {
        background:
            radial-gradient(circle at 0% 0%, rgba(233, 196, 106, 0.28), transparent 40%),
            radial-gradient(circle at 100% 100%, rgba(42, 157, 143, 0.18), transparent 38%),
            var(--bg-card) !important;
    }

    .hero-shell {
        background:
            linear-gradient(120deg, rgba(42, 157, 143, 0.13), rgba(233, 196, 106, 0.16), rgba(231, 111, 81, 0.1)),
            var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 18px !important;
        padding: 28px !important;
        margin-bottom: 20px !important;
        box-shadow: var(--card-shadow) !important;
        animation: fadeInUp 0.45s ease-out !important;
    }

    .hero-title {
        font-size: 2rem !important;
        letter-spacing: 0.3px !important;
        margin: 0 !important;
        color: var(--brand-ink) !important;
        font-family: "Fraunces", Georgia, serif !important;
        font-weight: 700 !important;
    }

    .hero-sub {
        margin-top: 8px !important;
        color: var(--text-secondary) !important;
        font-size: 1rem !important;
    }

    .stat-pills {
        display: flex !important;
        gap: 10px !important;
        flex-wrap: wrap !important;
        margin-top: 16px !important;
    }

    .stat-pill {
        background: rgba(255, 249, 238, 0.85) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 999px !important;
        padding: 7px 12px !important;
        font-size: 0.85rem !important;
        color: var(--text-primary) !important;
    }

    .filter-box {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 18px !important;
        margin-bottom: 14px !important;
    }

    .active-chip-row {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        margin: 4px 0 18px !important;
    }

    .active-chip {
        background: rgba(42, 157, 143, 0.12) !important;
        border: 1px solid rgba(42, 157, 143, 0.38) !important;
        color: #1b6e63 !important;
        border-radius: 999px !important;
        padding: 6px 10px !important;
        font-size: 0.8rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background: #f2e9da !important;
        border-radius: 10px !important;
        padding: 5px !important;
        border: 1px solid var(--border-color) !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(42, 157, 143, 0.2) !important;
        color: #1f6f66 !important;
    }

    .landing-hero {
        background:
            linear-gradient(130deg, rgba(42, 157, 143, 0.15), rgba(233, 196, 106, 0.2), rgba(231, 111, 81, 0.12)),
            #fffdf8 !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 18px !important;
        padding: 26px !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 18px !important;
    }

    .landing-hero h2 {
        margin: 0 !important;
        color: var(--brand-ink) !important;
        font-family: "Fraunces", Georgia, serif !important;
        font-size: 2.05rem !important;
    }

    .landing-hero p {
        margin: 10px 0 0 0 !important;
        color: var(--text-secondary) !important;
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }

    .social-strip {
        background: #fffaf0 !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        margin: 14px 0 16px 0 !important;
        color: var(--text-primary) !important;
        font-size: 0.9rem !important;
    }

    .sample-itinerary {
        background: #fffdf8 !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        box-shadow: var(--card-shadow) !important;
        margin-top: 8px !important;
    }

    .sample-day {
        background: #fff9ee !important;
        border: 1px solid #e6d8be !important;
        border-radius: 10px !important;
        padding: 10px 12px !important;
        margin-top: 8px !important;
        color: var(--text-primary) !important;
        font-size: 0.9rem !important;
    }

    .section-card {
        background: #fffdf8 !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 14px !important;
        margin-bottom: 10px !important;
        min-height: 96px !important;
    }

    .section-card h4 {
        margin: 0 0 6px 0 !important;
        color: var(--brand-ink) !important;
        font-size: 1rem !important;
    }

    .section-card p {
        margin: 0 !important;
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }

    .testimonial-card {
        background: #fffaf2 !important;
        border: 1px solid #e2d2b5 !important;
        border-radius: 12px !important;
        padding: 14px !important;
        height: 100% !important;
    }

    .faq-shell {
        background: #fffdf8 !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-top: 8px !important;
    }

    .welcome-shell {
        max-width: 980px !important;
        margin: 48px auto 0 auto !important;
        display: grid !important;
        grid-template-columns: 1.2fr 0.9fr !important;
        gap: 22px !important;
        align-items: stretch !important;
    }

    .welcome-hero {
        background:
            radial-gradient(circle at top left, rgba(233, 196, 106, 0.24), transparent 34%),
            radial-gradient(circle at bottom right, rgba(42, 157, 143, 0.18), transparent 30%),
            linear-gradient(135deg, #fffaf2, #f8f1e3) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 18px !important;
        padding: 34px !important;
        box-shadow: 0 14px 36px rgba(118, 95, 62, 0.12) !important;
    }

    .welcome-kicker {
        display: inline-block !important;
        padding: 6px 12px !important;
        border-radius: 999px !important;
        background: rgba(42, 157, 143, 0.12) !important;
        color: #1f6f66 !important;
        border: 1px solid rgba(42, 157, 143, 0.22) !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
    }

    .welcome-hero h1 {
        margin: 18px 0 12px 0 !important;
        color: var(--brand-ink) !important;
        font-size: 2.35rem !important;
        line-height: 1.1 !important;
        font-family: "Fraunces", Georgia, serif !important;
    }

    .welcome-hero p {
        margin: 0 !important;
        color: var(--text-secondary) !important;
        font-size: 1rem !important;
        line-height: 1.65 !important;
    }

    .welcome-highlights {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 10px !important;
        margin-top: 18px !important;
    }

    .welcome-highlight {
        background: rgba(255, 253, 248, 0.82) !important;
        border: 1px solid rgba(216, 204, 182, 0.9) !important;
        border-radius: 12px !important;
        padding: 10px 12px !important;
        font-size: 0.88rem !important;
        color: var(--text-primary) !important;
        min-width: 170px !important;
    }

    .welcome-highlight strong {
        display: block !important;
        color: var(--brand-ink) !important;
        margin-bottom: 3px !important;
    }

    .welcome-form-card {
        background: #fffdf8 !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 18px !important;
        padding: 28px !important;
        box-shadow: 0 14px 36px rgba(118, 95, 62, 0.1) !important;
    }

    .welcome-form-card h3 {
        margin: 0 0 8px 0 !important;
        color: var(--brand-ink) !important;
        font-family: "Fraunces", Georgia, serif !important;
        font-size: 1.4rem !important;
    }

    .welcome-form-card p {
        margin: 0 0 14px 0 !important;
        color: var(--text-secondary) !important;
        font-size: 0.95rem !important;
        line-height: 1.55 !important;
    }

    .welcome-note {
        margin-top: 14px !important;
        padding: 10px 12px !important;
        border-radius: 12px !important;
        background: #fff8ea !important;
        border: 1px solid #ead9b9 !important;
        color: #675a44 !important;
        font-size: 0.86rem !important;
    }

    .welcome-actions {
        margin-top: 6px !important;
    }

    .welcome-actions .stButton > button {
        min-height: 42px !important;
        height: 42px !important;
        padding: 0.55rem 0.9rem !important;
        border-radius: 10px !important;
    }

    div[data-testid="stPills"] [role="radiogroup"] label,
    div[data-testid="stPills"] [role="radio"],
    div[data-testid="stPills"] button[role="radio"] {
        border-radius: 999px !important;
        border: 1px solid rgba(216, 204, 182, 0.95) !important;
        background: #f6eee0 !important;
        color: #4b4235 !important;
        font-weight: 600 !important;
        padding: 0.28rem 0.7rem !important;
        opacity: 1 !important;
    }

    div[data-testid="stPills"] [role="radiogroup"] label *,
    div[data-testid="stPills"] [role="radio"] *,
    div[data-testid="stPills"] button[role="radio"] * {
        color: #4b4235 !important;
        fill: #4b4235 !important;
    }

    div[data-testid="stPills"] [role="radiogroup"] label:hover,
    div[data-testid="stPills"] [role="radio"]:hover,
    div[data-testid="stPills"] button[role="radio"]:hover {
        background: #ece3d3 !important;
        border-color: rgba(164, 142, 108, 0.85) !important;
        color: #3f372c !important;
    }

    div[data-testid="stPills"] [role="radiogroup"] label:hover *,
    div[data-testid="stPills"] [role="radio"]:hover *,
    div[data-testid="stPills"] button[role="radio"]:hover * {
        color: #3f372c !important;
        fill: #3f372c !important;
    }

    div[data-testid="stPills"] [role="radiogroup"] input:checked + div,
    div[data-testid="stPills"] [role="radiogroup"] label:has(input:checked),
    div[data-testid="stPills"] [role="radio"][aria-checked="true"],
    div[data-testid="stPills"] button[role="radio"][aria-checked="true"] {
        background: rgba(42, 157, 143, 0.2) !important;
        border-color: rgba(42, 157, 143, 0.75) !important;
        color: #155f56 !important;
        box-shadow: 0 6px 14px rgba(42, 157, 143, 0.16) !important;
    }

    div[data-testid="stPills"] [role="radiogroup"] input:checked + div *,
    div[data-testid="stPills"] [role="radiogroup"] label:has(input:checked) *,
    div[data-testid="stPills"] [role="radio"][aria-checked="true"] *,
    div[data-testid="stPills"] button[role="radio"][aria-checked="true"] * {
        color: #155f56 !important;
        fill: #155f56 !important;
    }

    .current-vibe {
        margin-top: 8px !important;
        display: inline-block !important;
        padding: 6px 12px !important;
        border-radius: 999px !important;
        background: rgba(42, 157, 143, 0.12) !important;
        color: #155f56 !important;
        border: 1px solid rgba(42, 157, 143, 0.42) !important;
        font-size: 0.84rem !important;
        font-weight: 700 !important;
    }

    /* Explore-by-vibe radio chips */
    div[data-testid="stRadio"] [role="radiogroup"] {
        gap: 8px !important;
        flex-wrap: wrap !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label {
        border-radius: 999px !important;
        border: 1px solid #d2c6b1 !important;
        background: #f5eee2 !important;
        padding: 0.26rem 0.7rem !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label p {
        color: #4a4135 !important;
        font-weight: 700 !important;
    }

    /* 1st option is 'All styles' */
    div[data-testid="stRadio"] [role="radiogroup"] label:nth-child(1) {
        background: #f1eadf !important;
        border-color: #d6c8b1 !important;
    }

    /* 2nd: Beaches */
    div[data-testid="stRadio"] [role="radiogroup"] label:nth-child(2) {
        background: #e5f6fa !important;
        border-color: #8dc8d6 !important;
    }

    /* 3rd: Temples */
    div[data-testid="stRadio"] [role="radiogroup"] label:nth-child(3) {
        background: #f7ece1 !important;
        border-color: #d8b083 !important;
    }

    /* 4th: Forts */
    div[data-testid="stRadio"] [role="radiogroup"] label:nth-child(4) {
        background: #efe9fb !important;
        border-color: #b8a7e0 !important;
    }

    /* 5th: Nature */
    div[data-testid="stRadio"] [role="radiogroup"] label:nth-child(5) {
        background: #e9f6e9 !important;
        border-color: #9ecf9e !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        box-shadow: 0 0 0 1px rgba(47, 42, 34, 0.24) inset !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
        color: #2f2a22 !important;
    }

    /* Make filter labels clearly visible */
    div[data-testid="stWidgetLabel"] p,
    .stSelectbox label,
    .stTextInput label,
    .stNumberInput label,
    .stMultiSelect label {
        color: #2f2a22 !important;
        font-weight: 800 !important;
        letter-spacing: 0.2px !important;
        opacity: 1 !important;
    }

    details summary p {
        color: #2f2a22 !important;
        font-weight: 700 !important;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
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
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .stApp {
            padding: 0.5rem !important;
        }

        .main .block-container {
            padding: 0.75rem !important;
            border-radius: 12px !important;
        }

        .hero-shell {
            padding: 18px !important;
            border-radius: 10px !important;
        }

        .hero-title {
            font-size: 1.45rem !important;
            line-height: 1.25 !important;
        }

        .hero-sub {
            font-size: 0.92rem !important;
        }

        .stat-pill {
            font-size: 0.78rem !important;
            padding: 6px 9px !important;
        }

        .filter-box {
            padding: 12px !important;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.86rem !important;
            padding: 6px 8px !important;
        }
        
        .search-hero {
            padding: 20px !important;
            margin-bottom: 20px !important;
        }
        
        .search-hero h2 {
            font-size: 1.3rem !important;
        }
        
        .place-card {
            margin-bottom: 16px !important;
        }
        
        .place-card .img {
            height: 170px !important;
        }
        
        .place-card .body {
            padding: 12px !important;
        }
        
        .place-card h3 {
            font-size: 1.05rem !important;
        }
        
        .btn {
            padding: 8px 12px !important;
            font-size: 0.85rem !important;
        }

        .landing-hero {
            padding: 18px !important;
        }

        .landing-hero h2 {
            font-size: 1.45rem !important;
            line-height: 1.25 !important;
        }

        .welcome-shell {
            grid-template-columns: 1fr !important;
            margin-top: 22px !important;
        }

        .welcome-hero,
        .welcome-form-card {
            padding: 20px !important;
        }

        .welcome-hero h1 {
            font-size: 1.7rem !important;
        }

        .stButton > button {
            border-radius: 9px !important;
            padding: 0.55rem 0.75rem !important;
            font-size: 0.84rem !important;
        }
        
        .results-summary {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 8px !important;
        }
    }
    
    /* Extra small mobile devices */
    @media (max-width: 480px) {
        .hero-title {
            font-size: 1.25rem !important;
        }

        .hero-sub {
            font-size: 0.86rem !important;
        }

        .search-hero h2 {
            font-size: 1.2rem !important;
        }
        
        .place-card .img {
            height: 150px !important;
        }
        
        .place-card h3 {
            font-size: 1rem !important;
        }
        
        .btn {
            padding: 6px 10px !important;
            font-size: 0.8rem !important;
        }
    }
    
    /* Hide Streamlit menu and sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Full width layout */
    .main .block-container {
        max-width: 100% !important;
        padding: 2rem 3rem !important;
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.75rem !important;
        }
    }
    </style>
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
    """Legacy function for placeholder images - kept for backward compatibility"""
    images = [
        "https://picsum.photos/800/600?random=1",
        "https://picsum.photos/800/600?random=2", 
        "https://picsum.photos/800/600?random=3",
        "https://picsum.photos/800/600?random=4",
        "https://picsum.photos/800/600?random=5",
    ]
    return images[:n]

# (removed external API code)

def check_local_image(place_name: str) -> str:
    """
    Checks if a local image file exists for the given place name.
    Returns the image file path if found.
    
    Args:
        place_name: Name of the tourist place
        
    Returns:
        Local file path if found, otherwise returns None
        
    Example:
        If place_name is "Taj Mahal", checks for:
        - images/taj_mahal.jpg
        - images/taj_mahal.png
        - images/Taj_Mahal.jpg
        etc.
    """
    # Create images directory path relative to the script
    images_dir = Path(__file__).parent / "images"
    
    # Normalize place name for filename (lowercase, replace spaces with underscores)
    normalized_name = place_name.lower().replace(" ", "_").replace("-", "_")
    
    # List of image extensions to check
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    
    # Check for normalized filename variations
    for ext in image_extensions:
        # Try normalized lowercase
        image_path = images_dir / f"{normalized_name}{ext}"
        if image_path.exists():
            # Return as string path for Streamlit compatibility
            return str(image_path)
        
        # Try with original case
        original_name = place_name.replace(" ", "_").replace("-", "_")
        image_path = images_dir / f"{original_name}{ext}"
        if image_path.exists():
            return str(image_path)
    
    return None

def convert_local_image_to_data_uri(image_path: str) -> str:
    """
    Converts a local image file to a base64 data URI for embedding in HTML.
    
    This is necessary because HTML img tags in Streamlit can't directly reference
    local file paths - they need data URIs or URLs.
    
    Args:
        image_path: Path to the local image file
        
    Returns:
        Base64 data URI string (e.g., "data:image/jpeg;base64,...")
    """
    try:
        # Determine MIME type based on file extension
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        # Read image file and encode to base64
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:{mime_type};base64,{base64_data}"
    except Exception:
        # If conversion fails, return None to fall back to other options
        return None

@st.cache_data(ttl=3600)  # Cache image URLs for 1 hour
def get_image_url(place_name: str, city_name: str = None) -> str:
    """
    Hybrid image fetching function: Checks local images first, then Unsplash API.
    
    This function implements a smart image loading strategy:
    1. First checks for local image files in /images/ folder
    2. Falls back to a default placeholder if not found
    
    Args:
        place_name: Name of the tourist place (primary search term)
        city_name: Optional city name to enhance search (not currently used but available)
        
    Returns:
        Image URL string (local path or remote URL)
        
    Example Usage:
        img_url = get_image_url("Taj Mahal", "Agra")
        st.image(img_url, caption="Taj Mahal")
    """
    # Default placeholder image URL using Picsum Photos with a deterministic seed
    seed_slug = place_name.lower().replace(" ", "-")
    default_placeholder = f"https://picsum.photos/seed/{quote(seed_slug)}/800/600"
    
    # Step 1: Check for local image file first (fastest, no API calls)
    local_image_path = check_local_image(place_name)
    if local_image_path:
        # Convert local image to data URI for HTML embedding
        data_uri = convert_local_image_to_data_uri(local_image_path)
        if data_uri:
            return data_uri
        # If conversion fails, fall through to next step
    
    # Step 2: Fallback to default placeholder
    return default_placeholder

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    # try loading dataset from expected path
    try:
        data_dir = Path(__file__).parent
        df = pd.read_csv(data_dir / "Touristnew.csv")
        d = pd.read_csv(data_dir / "Tourist5.csv")
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
if "name_input" not in st.session_state:
    st.session_state.name_input = ""
if "clear_name_input" not in st.session_state:
    st.session_state.clear_name_input = False

# ---------------------------
# Name Input (if not set) - TripAdvisor Style
# ---------------------------
if st.session_state.user_name is None:
    if st.session_state.clear_name_input:
        st.session_state.name_input = ""
        st.session_state.clear_name_input = False

    outer_left, outer_center, outer_right = st.columns([0.12, 1, 0.12])
    with outer_center:
        hero_col, form_col = st.columns([1.25, 0.95], gap="large")

    with hero_col:
        st.markdown(
            """
            <div class="welcome-hero">
                <span class="welcome-kicker">Smart destination discovery</span>
                <h1>Find places that actually match your travel style.</h1>
                <p>Horizon Atlas helps you explore better destinations faster with focused recommendations, filters that make sense, and a shortlist you can actually use.</p>
                <div class="welcome-highlights">
                    <div class="welcome-highlight">
                        <strong>Personalized picks</strong>
                        Search by destination, style, and budget.
                    </div>
                    <div class="welcome-highlight">
                        <strong>Shortlist faster</strong>
                        Save the places worth comparing.
                    </div>
                    <div class="welcome-highlight">
                        <strong>Ready to act</strong>
                        Open locations directly in Google Maps.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        with st.container(border=True):
            st.markdown("### Start with your name")
            st.caption("Personalize your experience now, or continue as a guest and explore recommendations right away.")

            user_name = st.text_input(
                "Your Name",
                placeholder="Enter your name here",
                key="name_input",
                label_visibility="visible",
                max_chars=40,
            )

            with st.container():
                st.markdown('<div class="welcome-actions">', unsafe_allow_html=True)
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    continue_clicked = st.button("Continue", type="primary", use_container_width=True, key="welcome_continue_btn")
                with btn_col2:
                    skip_clicked = st.button("Skip", use_container_width=True, key="welcome_skip_btn")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="welcome-note">
                    Explore <strong>{int(df['City'].nunique())}</strong> cities and <strong>{int(df['Place'].nunique())}</strong> destinations without any setup.
                </div>
                """,
                unsafe_allow_html=True,
            )

        if continue_clicked:
            cleaned_name = " ".join(user_name.split())
            if cleaned_name:
                st.session_state.user_name = cleaned_name
                st.session_state.clear_name_input = True
                st.rerun()
            st.error("Please enter a valid name.")

        if skip_clicked:
            st.session_state.user_name = "Guest"
            st.session_state.clear_name_input = True
            st.rerun()
    
    st.stop()  # Stop execution until name is entered

# ---------------------------
# Render helper
# ---------------------------
def render_destination_cards(dataframe: pd.DataFrame, key_prefix: str = "explore", max_results: int = 12, cols_num: int = 3):
    cols_num = max(1, int(cols_num))
    display_results = dataframe.head(max_results)

    if len(display_results) == 0:
        st.info("No results match your search right now. Try clearing filters or using a broader keyword.")
        return

    rows = int(np.ceil(len(display_results) / cols_num))
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_num, gap="large")
        for col in cols:
            if idx >= len(display_results):
                col.empty()
                idx += 1
                continue

            row = display_results.iloc[idx]
            place_name = row.get("Place", "Unknown Place")
            city_name = row.get("City", "Unknown City")
            desc = str(row.get("Place_desc", ""))[:200]
            rating = row.get("Ratings_x", None)
            distance = row.get("Distance", "-")
            best_time = row.get("Best_time_to_visit", "-")
            budget = row.get("Budget_Range", "-")
            travel_type = row.get("Travel_Type", "-")
            category = row.get("Category", "Explore")

            img_url = get_image_url(place_name, city_name)

            place_name_html = escape(str(place_name))
            city_name_html = escape(str(city_name))
            desc_html = escape(desc)
            distance_html = escape(str(distance))
            best_time_html = escape(str(best_time))
            budget_html = escape(str(budget))
            travel_type_html = escape(str(travel_type))
            category_html = escape(str(category))
            img_url_html = escape(str(img_url), quote=True)
            maps_url = get_google_maps_url(place_name, city_name)
            maps_url_html = escape(maps_url, quote=True)

            rating_html = ""
            if pd.notna(rating):
                rating_html = f'<span class="rating-badge"><span class="rating-value">{escape(str(rating))}</span> ⭐</span>'

            card_html = f"""
            <div class="place-card">
                <div class="image-wrap">
                    <img src="{img_url_html}" class="img" alt="{place_name_html}" loading="lazy" onerror="this.src='https://picsum.photos/seed/{quote(str(place_name).replace(' ', '-'))}/800/600'">
                    <div class="image-overlay">
                        <span class="category-pill">{category_html}</span>
                    </div>
                </div>
                <div class="body">
                    <h3><a class="place-title-link" href="{maps_url_html}" target="_blank" title="Open in Google Maps">{place_name_html}</a>{rating_html}</h3>
                    <div class="location-info">📍 {city_name_html} · {distance_html} · Best time: {best_time_html}</div>
                    <p>{desc_html}{"..." if len(desc) >= 200 else ""}</p>
                    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #d8ccb6; display: flex; gap: 8px; flex-wrap: wrap;">
                        <span style="font-size: 0.85rem; color: #61584b;">💰 {budget_html}</span>
                        <span style="font-size: 0.85rem; color: #61584b;">👥 {travel_type_html}</span>
                    </div>
                </div>
            </div>
            """

            col.markdown(card_html, unsafe_allow_html=True)

            like_key = f"{key_prefix}_like__{idx}__{place_name}"
            dislike_key = f"{key_prefix}_dislike__{idx}__{place_name}"

            with col:
                col_a, col_b, col_c = st.columns([0.9, 0.9, 1.8])
                with col_a:
                    like_text = "Saved" if place_name in st.session_state.liked_places else "Save"
                    if st.button(like_text, key=like_key):
                        st.session_state.liked_places.add(place_name)
                        st.session_state.disliked_places.discard(place_name)
                        st.toast(f"Saved {place_name}")
                        st.rerun()
                with col_b:
                    dislike_text = "Hidden" if place_name in st.session_state.disliked_places else "Hide"
                    if st.button(dislike_text, key=dislike_key):
                        st.session_state.disliked_places.add(place_name)
                        st.session_state.liked_places.discard(place_name)
                        st.toast(f"Hidden {place_name}")
                        st.rerun()
                with col_c:
                    st.markdown(
                        f'<a class="btn btn-primary card-map-btn" href="{maps_url_html}" target="_blank">🗺️ Open Maps</a>',
                        unsafe_allow_html=True,
                    )
            idx += 1


# ---------------------------
# Header, hero, search and tabs
# ---------------------------
scroll_anchor = st.empty()
scroll_anchor.markdown('<div id="top"></div>', unsafe_allow_html=True)

if "main_search" not in st.session_state:
    st.session_state.main_search = ""
if "city_filter" not in st.session_state:
    st.session_state.city_filter = "Any"
if "category_filter" not in st.session_state:
    st.session_state.category_filter = "Any"
if "budget_filter" not in st.session_state:
    st.session_state.budget_filter = "Any"
if "duration_filter" not in st.session_state:
    st.session_state.duration_filter = "Any"
if "travel_filter" not in st.session_state:
    st.session_state.travel_filter = "Any"
if "surprise_pick" not in st.session_state:
    st.session_state.surprise_pick = ""
if "card_columns" not in st.session_state:
    st.session_state.card_columns = 3
if "pending_main_search" not in st.session_state:
    st.session_state.pending_main_search = None
if "pending_filter_updates" not in st.session_state:
    st.session_state.pending_filter_updates = None
if "scroll_to_recommendations" not in st.session_state:
    st.session_state.scroll_to_recommendations = False
if "popular_style_choice" not in st.session_state:
    st.session_state.popular_style_choice = None

# Apply queued search updates before rendering the text input widget.
if st.session_state.pending_main_search is not None:
    st.session_state.main_search = st.session_state.pending_main_search
    st.session_state.pending_main_search = None

# Apply queued filter updates before rendering filter widgets.
if st.session_state.pending_filter_updates is not None:
    for key, value in st.session_state.pending_filter_updates.items():
        st.session_state[key] = value
    st.session_state.pending_filter_updates = None

st.markdown(
    """
    <div class="header-nav">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 20px; display: flex; align-items: center; justify-content: space-between; gap: 8px;">
            <h1>🧭 Horizon Atlas</h1>
            <div style="font-size: 0.9rem; color: #61584b;">Filter smart. Explore better.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_user1, col_user2, col_user3 = st.columns([2, 1, 1])
with col_user1:
    st.markdown(f"**Welcome, {st.session_state.user_name}**")
with col_user2:
    if st.button("🔄 Change Name", use_container_width=True):
        st.session_state.user_name = None
        st.session_state.clear_name_input = True
        st.rerun()
with col_user3:
    st.metric("❤️ Saved", len(st.session_state.liked_places))

city_count = int(df["City"].nunique()) if "City" in df.columns else 0
place_count = int(df["Place"].nunique()) if "Place" in df.columns else 0
avg_rating = round(pd.to_numeric(df.get("Ratings_x", pd.Series(dtype=float)), errors="coerce").dropna().mean(), 2) if "Ratings_x" in df.columns else 0

st.markdown(
    f"""
    <div class="landing-hero">
        <h2>Find Better Destinations in Minutes</h2>
        <p>Personalized destination suggestions and smart recommendations based on your style and budget.</p>
        <div class="stat-pills">
            <span class="stat-pill">🏙️ {city_count} cities</span>
            <span class="stat-pill">📍 {place_count} destinations</span>
            <span class="stat-pill">⭐ Avg rating {avg_rating}</span>
            <span class="stat-pill">🧳 {len(df)} itinerary-ready picks</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="social-strip">✅ One simple recommender · ✅ Smart suggestions · ✅ Save favorites and export recommendations</div>', unsafe_allow_html=True)

st.markdown('<div class="filter-box">', unsafe_allow_html=True)
st.markdown("### Get Recommendations")

quick_col1, quick_col2, quick_col3 = st.columns([2, 1, 1])
with quick_col1:
    search_text = st.text_input(
        "Destination or keyword",
        placeholder="Try Taj Mahal, Goa beaches, adventure, family...",
        key="main_search",
    )
with quick_col2:
    duration_filter = st.selectbox("Days", options=["Any", "1-2 days", "3-5 days", "1 week+"], key="duration_filter")
with quick_col3:
    travel_type_filter = st.selectbox("Style", options=["Any", "Solo", "Couple", "Family", "Group"], key="travel_filter")

quick_categories = sorted(df["Category"].dropna().unique().tolist())
if quick_categories:
    st.caption("Explore by vibe")

    category_icons = {
        "Beach": "🏖",
        "Beaches": "🏖",
        "Fort": "🏰",
        "Forts": "🏰",
        "Nature": "🌿",
        "Temple": "🛕",
        "Temples": "🛕",
        "Adventure": "🧗",
        "Heritage": "🏛",
        "Hills": "⛰",
        "Nightlife": "🌃",
        "Wildlife": "🦜",
    }

    def get_style_label(category_name: str) -> str:
        icon = category_icons.get(category_name, "✨")
        count = int((df["Category"] == category_name).sum())
        pretty_name = "Hidden Gems" if category_name.lower() == "other" else category_name
        return f"{icon} {pretty_name} · {count}"

    preferred_order = [
        "Beaches", "Beach",
        "Temple", "Temples",
        "Fort", "Forts",
        "Nature",
    ]

    top_categories = []
    for category in preferred_order:
        if category in quick_categories and category not in top_categories:
            top_categories.append(category)

    for category in quick_categories:
        if category not in top_categories:
            top_categories.append(category)
        if len(top_categories) >= 5:
            break

    vibe_options = ["Any"] + top_categories
    if st.session_state.category_filter in top_categories:
        default_index = vibe_options.index(st.session_state.category_filter)
    else:
        default_index = 0

    selected_vibe = st.radio(
        "Popular picks",
        options=vibe_options,
        index=default_index,
        horizontal=True,
        key="popular_style_choice_radio",
        format_func=lambda x: "All styles" if x == "Any" else get_style_label(x),
        label_visibility="collapsed",
    )

    desired_category_filter = selected_vibe if selected_vibe != "Any" else "Any"
    if st.session_state.category_filter != desired_category_filter:
        st.session_state.pending_filter_updates = {"category_filter": desired_category_filter}
        st.session_state.scroll_to_recommendations = True
        st.rerun()


action_col1, action_col2, action_col3 = st.columns(3)
with action_col1:
    if st.button("✨ Show Recommendations", use_container_width=True):
        st.session_state.scroll_to_recommendations = True
        st.rerun()
with action_col2:
    if st.button("🎲 Surprise Me", use_container_width=True):
        if not df.empty:
            pick = df.sample(1, random_state=None).iloc[0]
            st.session_state.surprise_pick = str(pick.get("Place", ""))
            st.session_state.pending_main_search = st.session_state.surprise_pick
            st.session_state.pending_filter_updates = {
                "city_filter": "Any",
                "category_filter": "Any",
                "budget_filter": "Any",
                "duration_filter": "Any",
                "travel_filter": "Any",
            }
            st.session_state.scroll_to_recommendations = True
            st.rerun()
with action_col3:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.pending_main_search = ""
        st.session_state.pending_filter_updates = {
            "city_filter": "Any",
            "category_filter": "Any",
            "budget_filter": "Any",
            "duration_filter": "Any",
            "travel_filter": "Any",
        }
        st.rerun()

with st.expander("Advanced filters"):
    cities = sorted(df["City"].dropna().unique().tolist())
    categories = sorted(df["Category"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        city_filter = st.selectbox("📍 City", options=["Any"] + cities, key="city_filter")
    with col2:
        category_filter = st.selectbox("🏷️ Category", options=["Any"] + categories, key="category_filter")

    col3, col4 = st.columns(2)
    with col3:
        budget_filter = st.selectbox("💰 Budget", options=["Any", "Low (<₹5000)", "Medium (₹5000-15000)", "High (>₹15000)"], key="budget_filter")
    with col4:
        card_columns = st.selectbox("🧱 Card Columns", options=[1, 2, 3], key="card_columns")

# Keep values available even when the expander is closed.
city_filter = st.session_state.city_filter
category_filter = st.session_state.category_filter
budget_filter = st.session_state.budget_filter
card_columns = st.session_state.card_columns

st.markdown('</div>', unsafe_allow_html=True)

results = df.copy()
q = search_text.strip().lower()
if q:
    mask = (
        results["Place"].str.lower().str.contains(q, na=False)
        | results["City"].str.lower().str.contains(q, na=False)
        | results["Category"].str.lower().str.contains(q, na=False)
    )
    results = results[mask]

filters = {
    "City": city_filter,
    "Category": category_filter,
    "Budget_Range": budget_filter,
    "Duration": duration_filter,
    "Travel_Type": travel_type_filter,
}
for column_name, selected_value in filters.items():
    if selected_value != "Any":
        results = results[results[column_name] == selected_value]

if "Ratings_x" in results.columns:
    results = results.sort_values(by="Ratings_x", ascending=False)

active_filters = []
if q:
    active_filters.append(f"Search: {escape(q)}")
for column_name, selected_value in filters.items():
    if selected_value != "Any":
        active_filters.append(f"{column_name}: {escape(str(selected_value))}")

if active_filters:
    chips_html = "".join([f'<span class="active-chip">{item}</span>' for item in active_filters])
    st.markdown(f'<div class="active-chip-row">{chips_html}</div>', unsafe_allow_html=True)

st.markdown('<div id="recommendations"></div>', unsafe_allow_html=True)
if st.session_state.scroll_to_recommendations:
    components.html(
        """
        <script>
            const scrollToRecommendations = () => {
                const target = window.parent.document.getElementById('recommendations');
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            };
            setTimeout(scrollToRecommendations, 120);
        </script>
        """,
        height=0,
    )
    st.session_state.scroll_to_recommendations = False

tab_explore, tab_favorites, tab_insights = st.tabs(["🌍 Explore", "❤️ Favorites", "📊 Insights"])

with tab_explore:
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown(f"**Found {len(results)} destinations**")
    with col_right:
        st.markdown(f"❤️ **{len(st.session_state.liked_places)} saved**")

    render_destination_cards(results, key_prefix="explore", max_results=12, cols_num=card_columns)

    if not results.empty:
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download current recommendations (CSV)",
            data=csv_bytes,
            file_name="recommendations.csv",
            mime="text/csv",
        )

with tab_favorites:
    liked_set = set(st.session_state.liked_places)
    if liked_set:
        favorites_df = df[df["Place"].isin(liked_set)].copy()
        if "Ratings_x" in favorites_df.columns:
            favorites_df = favorites_df.sort_values(by="Ratings_x", ascending=False)
        st.markdown(f"### Your shortlist ({len(favorites_df)})")
        render_destination_cards(favorites_df, key_prefix="fav", max_results=24, cols_num=card_columns)
    else:
        st.info("Save destinations in Explore to build your shortlist here.")

with tab_insights:
    st.markdown("### Your preference snapshot")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Results Now", len(results))
    with m2:
        st.metric("Liked Places", len(st.session_state.liked_places))
    with m3:
        st.metric("Hidden Places", len(st.session_state.disliked_places))

    if st.session_state.liked_places or st.session_state.disliked_places:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 👍 Liked")
            for place in sorted(st.session_state.liked_places):
                st.markdown(f"- {place}")
        with c2:
            st.markdown("#### 👎 Hidden")
            for place in sorted(st.session_state.disliked_places):
                st.markdown(f"- {place}")
    else:
        st.info("Interact with destinations in Explore to build preference insights.")

st.markdown("### Travelers Say")
test_col1, test_col2, test_col3 = st.columns(3)
with test_col1:
    st.markdown('<div class="testimonial-card"><strong>\"Choosing got 10x faster\"</strong><p style="margin-top:8px;">I stopped juggling tabs and built a clean shortlist in minutes.</p></div>', unsafe_allow_html=True)
with test_col2:
    st.markdown('<div class="testimonial-card"><strong>\"Great for family pace\"</strong><p style="margin-top:8px;">The suggestions balanced attractions and downtime for all ages.</p></div>', unsafe_allow_html=True)
with test_col3:
    st.markdown('<div class="testimonial-card"><strong>\"Useful budget filters\"</strong><p style="margin-top:8px;">I quickly narrowed places that matched our trip style and spend.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="faq-shell">', unsafe_allow_html=True)
st.markdown("### FAQs")
with st.expander("Is this free to use?"):
    st.write("Yes. You can search, filter, and shortlist places without any paid subscription in this app version.")
with st.expander("Can I customize itinerary style?"):
    st.write("Yes. Use travel style, budget, city, and category filters to tune recommendations.")
with st.expander("Can I save and share recommendations?"):
    st.write("You can save favorites in-app and download your current recommendation list as a CSV file.")
with st.expander("Does it work for solo, couple, and family trips?"):
    st.write("Yes. Select the travel type filter to get results that fit your trip context.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(42,157,143,0.16), rgba(233,196,106,0.18), rgba(231,111,81,0.12)); padding: 24px; border-radius: 14px; border: 1px solid #d8ccb6; margin: 8px 0 18px 0; text-align: center; box-shadow: 0 8px 24px rgba(118, 95, 62, 0.08);">
        <h3 style="margin:0; color:#2f2a22; font-family: 'Fraunces', Georgia, serif;">Ready to discover your next destination?</h3>
        <p style="margin:10px 0 0 0; color:#61584b;">Use the recommender above, save your favorites, and export results in one flow.</p>
    </div>
    <div style="background: linear-gradient(130deg, #fff9ee, #f9f0df); padding: 22px; border-radius: 12px; border: 1px solid #d8ccb6; margin-top: 24px; text-align: center; box-shadow: 0 8px 24px rgba(118, 95, 62, 0.08);">
        <p style="color: #61584b; margin: 0; font-size: 0.9rem;">
            Built with ❤️ by <strong style="color: #e76f51;">Shambhuraje Jadhav</strong> · Horizon Atlas Edition
        </p>
        <p style="color: #7a705e; margin: 8px 0 0 0; font-size: 0.85rem;">
            Save places, refine quickly, and choose better destinations in one flow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

