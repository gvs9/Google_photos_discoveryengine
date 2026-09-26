import streamlit as st
import os
import streamlit.components.v1 as components

# Set up the page configuration to take up the full screen
st.set_page_config(
    page_title="GPDE Telemetry Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit's default UI elements (header, footer, padding) so the custom UI takes over completely
st.markdown("""
    <style>
        .block-container { 
            padding-top: 0rem; 
            padding-bottom: 0rem; 
            padding-left: 0rem; 
            padding-right: 0rem; 
            max-width: 100%; 
        }
        header { visibility: hidden; }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        /* Remove default white background */
        .stApp { background-color: #05070e; }
    </style>
""", unsafe_allow_html=True)

# Determine the path to the rich HTML prototype
html_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "stitch_gpde", "code.html")

try:
    # Read the exact HTML you were hosting locally
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_data = f.read()
    
    # Render the custom HTML dashboard inside Streamlit
    components.html(html_data, height=1000, scrolling=True)

except Exception as e:
    st.error(f"Could not load the custom UI. Error: {e}")
