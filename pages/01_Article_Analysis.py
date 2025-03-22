import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
import openai
from datetime import datetime

# --- CONSTANTS & CONFIGURATION ---
IMG_PATH_LOGO_FULL = "img/logo_full.png"
IMG_PATH_LOGO_ICON = "img/logo_icon.png"
ARTICLES_CLEAN_DIR = "data/articles_clean"
METADATA_PATH = "data/metadata.csv"

# Default model settings
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

# --- INITIALIZATION ---
def initialize_page():
    """Initialize the page, load environment variables, and set up OpenAI client."""
    # Load environment variables
    load_dotenv()
    
    # Set up API keys
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    
    # Set up OpenAI client with OpenRouter
    client = openai.OpenAI(
        api_key=openrouter_api_key,
        base_url=openrouter_api_base
    )
    
    # Configure page
    st.set_page_config(
        page_title="SARA | Article Analysis", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=IMG_PATH_LOGO_ICON
    )
    
    # Initialize session state for this page
    if 'analysis_prompt' not in st.session_state:
        st.session_state.analysis_prompt = ""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = ""
    
    return client

# --- SIDEBAR SETUP ---
def setup_sidebar():
    """Set up the sidebar UI components."""
    with st.sidebar:
        st.logo(image=IMG_PATH_LOGO_FULL, link="https://nexus-group.ai", icon_image=IMG_PATH_LOGO_ICON, size="large")
        st.title("SARA")
        st.caption("Summarize. Analyze. Retrieve. Annotate.")
        st.markdown("---")

# --- MAIN PAGE FUNCTIONALITY ---
def main():
    """Main function to run the Article Analysis page."""
    # Initialize the page
    client = initialize_page()
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content
    st.title("Article Analysis")
    st.write("This page will provide advanced analysis features for Wiener Zeitung articles.")
    
    # Placeholder for future analysis features
    st.info("Analysis features are under development. Check back soon!")
    
    # Example UI elements for the future analysis page
    with st.expander("Planned Features"):
        st.markdown("""
        - Entity extraction and visualization
        - Sentiment analysis across time periods
        - Topic modeling and trend identification
        - Comparative analysis between articles
        - Historical context enrichment
        """)

if __name__ == "__main__":
    main()