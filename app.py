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
MAX_ARTICLES_TO_DISPLAY = 1000  # Safety limit

# --- INITIALIZATION ---
def initialize_app():
    """Initialize the app, load environment variables, and set up OpenAI client."""
    # Load environment variables
    load_dotenv()
    
    # Set up API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    
    # Set up OpenAI client with OpenRouter
    client = openai.OpenAI(
        api_key=openrouter_api_key,
        base_url=openrouter_api_base
    )
    
    # Configure page
    st.set_page_config(
        page_title="SARA | Nexus Group AI", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=IMG_PATH_LOGO_ICON
    )
    
    # Initialize session state
    if 'last_prompt' not in st.session_state:
        st.session_state.last_prompt = ""
    if 'last_model' not in st.session_state:
        st.session_state.last_model = ""
    if 'api_error' not in st.session_state:
        st.session_state.api_error = False
    
    return client

# --- DATA HANDLING FUNCTIONS ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_metadata():
    """Load and preprocess the metadata CSV file."""
    try:
        df = pd.read_csv(METADATA_PATH)
        # Convert date columns immediately to avoid repeated conversions
        df["published_at"] = pd.to_datetime(df["published_at"])
        return df
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_article(filename):
    """Load an article from its JSON file."""
    try:
        with open(os.path.join(ARTICLES_CLEAN_DIR, filename), "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading article: {e}")
        return None

def get_llm_response(client, prompt, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):
    """Get a response from the LLM model."""
    try:
        # Add debugging info
        st.session_state.last_prompt = prompt
        st.session_state.last_model = model
        
        # Determine desired language from prompt
        language_instruction = ""
        if "Write your response in English" in prompt:
            language_instruction = "You must respond in English."
        elif "Write your response in German" in prompt:
            language_instruction = "You must respond in German."
        
        system_prompt = f"You are SARA, an intelligent assistant for news article analysis. {language_instruction}"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in LLM call: {e}")
        # Add detailed error info
        if hasattr(e, 'response'):
            st.error(f"Response details: {e.response}")
        st.session_state.api_error = True
        return "Sorry, I encountered an error while processing your request. Please check the API configuration."

# --- PROMPT TEMPLATES ---
# --- UPDATED PROMPT TEMPLATES ---
def create_transformation_prompt(article_text, transformation_type, level=None, language="English"):
    """Create a prompt for text transformation based on the selected type, level, and language."""
    language_instruction = f"Write your response in {language}."
    
    prompts = {
        "simplify": f"""Transform the following news article into a simpler version at a {level} reading level. 
                     Maintain all key information and main points, but use simpler language and shorter sentences.
                     {language_instruction}
                     
                     Article: {article_text}""",
                     
        "conversational": f"""Rewrite the following news article in a conversational style, as if explaining it to a friend.
                           Keep all the important information but make it engaging and informal.
                           {language_instruction}
                           
                           Article: {article_text}""",
                           
        "academic": f"""Transform the following news article into an academic style with formal language.
                     Add depth to the analysis and use appropriate academic tone.
                     {language_instruction}
                     
                     Article: {article_text}""",
                     
        "youth": f"""Rewrite the following news article for a young audience (age 12-16).
                  Explain concepts clearly, use engaging language, but don't be condescending.
                  {language_instruction}
                  
                  Article: {article_text}"""
    }
    return prompts.get(transformation_type, "Invalid transformation type")

def create_summary_prompt(article_text, summary_type, length=None, language="English"):
    """Create a prompt for article summarization based on the selected type, length, and language."""
    language_instruction = f"Write your response in {language}."
    
    prompts = {
        "headline": f"""Create a single headline-style summary (10-15 words) of the following news article.
                      {language_instruction}
                      
                      Article: {article_text}""",
                      
        "paragraph": f"""Summarize the following news article in one paragraph (3-5 sentences).
                       {language_instruction}
                       
                       Article: {article_text}""",
                       
        "detailed": f"""Provide a detailed summary of the following news article in {length} paragraphs.
                      Include the most important information, key points, and any significant quotes or data.
                      {language_instruction}
                      
                      Article: {article_text}""",
                      
        "multi-perspective": f"""Analyze the following news article from multiple perspectives.
                              Identify different stakeholder viewpoints mentioned or implied in the article.
                              Present a balanced summary that acknowledges these different perspectives.
                              {language_instruction}
                              
                              Article: {article_text}"""
    }
    return prompts.get(summary_type, "Invalid summary type")

# --- UI COMPONENTS ---
def setup_sidebar():
    """Set up the sidebar UI components."""
    with st.sidebar:
        st.logo(image=IMG_PATH_LOGO_FULL, link="https://nexus-group.ai", icon_image=IMG_PATH_LOGO_ICON, size="large")
        st.title("SARA")
        st.caption("Summarize. Analyze. Retrieve. Annotate.")
        st.markdown("---")

def create_sidebar_filters():
    """Create and handle the sidebar filtering options, returning the filtered metadata."""
    with st.sidebar:
        st.subheader("Filter Articles")
        
        # Date range filter
        date_range = st.date_input(
            "Date range",
            [datetime(2010, 1, 1), datetime(2023, 6, 30)],
            format="YYYY-MM-DD",
        )
        
        # Load metadata for filtering
        metadata = load_metadata()
        if metadata.empty:
            st.error("No metadata available")
            return pd.DataFrame()
        
        # Filter by date
        start_date, end_date = date_range
        filtered_metadata = metadata[
            (metadata["published_at"].dt.date >= start_date) & 
            (metadata["published_at"].dt.date <= end_date)
        ].copy()
        
        # Convert timestamps to strings for display
        filtered_metadata['published_date_str'] = filtered_metadata['published_at'].dt.strftime('%Y-%m-%d')
        
        # Section filter
        sections = ["All"] + sorted(metadata["section"].unique().tolist())
        selected_section = st.selectbox("Section", sections)
        if selected_section != "All":
            filtered_metadata = filtered_metadata[filtered_metadata["section"] == selected_section]
        
        # Topic filter
        topics = ["All", "financial_crisis", "sustainability", "fake_news", "ai", 
                 "digitalization", "local_journalism", "covid", "demographics", "innovation"]
        selected_topic = st.selectbox("Topic", topics)
        if selected_topic != "All":
            # Filter by topic score > 0.7
            filtered_metadata = filtered_metadata[filtered_metadata[selected_topic] > 0.7]
        
        # Valid tags filter
        valid_only = st.checkbox("Only articles with valid tags", value=False)
        if valid_only:
            filtered_metadata = filtered_metadata[filtered_metadata["valid_indicator"] == True]
        
        st.info(f"Found {len(filtered_metadata)} matching articles")
        
        return filtered_metadata

def handle_article_selection(filtered_metadata):
    """Handle the article selection process based on filtered metadata."""
    if filtered_metadata.empty:
        return None
    
    with st.sidebar:
        st.subheader("Select an Article")
        
        # Check if we have too many articles
        too_many_articles = len(filtered_metadata) > MAX_ARTICLES_TO_DISPLAY
        
        if too_many_articles:
            filtered_metadata = apply_additional_filters(filtered_metadata)
        
        # Now show article selection if we have a reasonable number
        if len(filtered_metadata) > 0 and len(filtered_metadata) <= MAX_ARTICLES_TO_DISPLAY:
            # Display filtered articles
            selected_article_idx = st.selectbox(
                f"Choose from {len(filtered_metadata)} articles:", 
                range(len(filtered_metadata)),
                format_func=lambda x: f"{filtered_metadata.iloc[x]['published_date_str']} - {filtered_metadata.iloc[x]['title']}"
            )
            
            selected_article_filename = filtered_metadata.iloc[selected_article_idx]["filename"]
            return load_article(selected_article_filename), selected_article_filename
        else:
            if len(filtered_metadata) == 0:
                st.error("No articles match your current filters. Please adjust your criteria.")
            else:
                st.info(f"Still too many articles ({len(filtered_metadata)}). Please add more filters to narrow your selection.")
            return None, None

def apply_additional_filters(filtered_metadata):
    """Apply additional filters when too many articles are selected."""
    st.warning(f"Found {len(filtered_metadata)} articles. Please use additional filters to narrow down your selection.")
    
    # Additional filtering options to narrow down results
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by year
        years = sorted(filtered_metadata['published_at'].dt.year.unique())
        selected_year = st.selectbox("Select year", ["All"] + list(years))
        
        if selected_year != "All":
            filtered_metadata = filtered_metadata[filtered_metadata['published_at'].dt.year == selected_year]
    
    with col2:
        # Filter by month (only if year is selected)
        if selected_year != "All":
            months = sorted(filtered_metadata['published_at'].dt.month.unique())
            month_names = ["All"] + [f"{m} - {pd.Timestamp(2000, m, 1).strftime('%B')}" for m in months]
            selected_month_option = st.selectbox("Select month", month_names)
            
            if selected_month_option != "All":
                selected_month = int(selected_month_option.split(" - ")[0])
                filtered_metadata = filtered_metadata[filtered_metadata['published_at'].dt.month == selected_month]
    
    # After additional filtering, check if we still have too many articles
    if len(filtered_metadata) > MAX_ARTICLES_TO_DISPLAY:
        # Filter by title search
        title_search = st.text_input("Search in title", "")
        if title_search:
            filtered_metadata = filtered_metadata[filtered_metadata['title'].str.contains(title_search, case=False, na=False)]
    
    return filtered_metadata

def display_article(article_data):
    """Display the selected article content."""
    with st.expander("Original Article", expanded=True):
        st.subheader(article_data["title"])
        st.caption(f"Author: {article_data['author']} | Published: {article_data['published_at']}")
        st.caption(f"Category: {article_data['category']} | Section: {article_data['section']}")
        st.markdown(article_data["text"].replace("\n", "\n\n"))

def show_text_transformation_tab(client, article_data, selected_article_filename, desired_language):
    """Display the text transformation features tab with language support."""
    st.subheader("Adapt Reading Style")
    transformation_type = st.selectbox(
        "Select transformation type:",
        ["simplify", "conversational", "academic", "youth"],
        format_func=lambda x: x.capitalize()
    )
    
    level = None
    if transformation_type == "simplify":
        level = st.select_slider(
            "Reading level",
            options=["elementary", "middle school", "high school", "college"]
        )
    
    if st.button("Transform Text", key="transform_button"):
        with st.spinner(f"Transforming text in {desired_language}..."):
            try:
                prompt = create_transformation_prompt(
                    article_data["text"], 
                    transformation_type, 
                    level, 
                    language=desired_language
                )
                transformed_text = get_llm_response(client, prompt)
                st.markdown(transformed_text)
                
                # Add download button for transformed text
                st.download_button(
                    label="Download transformed text",
                    data=transformed_text,
                    file_name=f"transformed_{selected_article_filename.replace('.json', '')}_{desired_language.lower()}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error transforming text: {e}")
                st.session_state.api_error = True

def show_summarization_tab(client, article_data, selected_article_filename, desired_language):
    """Display the summarization features tab with language support."""
    st.subheader("Summarization")
    summary_type = st.selectbox(
        "Select summary type:",
        ["headline", "paragraph", "detailed", "multi-perspective"],
        format_func=lambda x: x.replace("-", " ").capitalize()
    )
    
    length = None
    if summary_type == "detailed":
        length = st.slider("Number of paragraphs", 1, 5, 2)
    
    if st.button("Generate Summary", key="summary_button"):
        with st.spinner(f"Generating {desired_language} summary..."):
            try:
                prompt = create_summary_prompt(
                    article_data["text"], 
                    summary_type, 
                    length, 
                    language=desired_language
                )
                summary = get_llm_response(client, prompt)
                st.markdown(summary)
                
                # Add download button for summary
                st.download_button(
                    label="Download summary",
                    data=summary,
                    file_name=f"summary_{selected_article_filename.replace('.json', '')}_{desired_language.lower()}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating summary: {e}")
                st.session_state.api_error = True

def show_troubleshooting_info():
    """Display API error troubleshooting information."""
    with st.expander("API Troubleshooting", expanded=True):
        st.subheader("API Error Detected")
        st.markdown("""
        ### Troubleshooting Steps:
        1. **Check your API key** - Make sure it's valid and has correct permissions
        2. **Verify the API endpoint** - Check if the OpenRouter API base URL is correct
        3. **Model availability** - Confirm the selected model is available through your API plan
        
        You can try entering your API key manually using the sidebar option.
        """)
        
        if st.session_state.last_prompt:
            st.subheader("Last Prompt Used")
            st.text_area("Prompt", st.session_state.last_prompt, height=200, disabled=True)
        
        if st.session_state.last_model:
            st.info(f"Last model used: {st.session_state.last_model}")
            
        if st.button("Clear Error State"):
            st.session_state.api_error = False
            st.rerun()

def display_language_indicator(desired_language):
    """Display a prominent indicator of the currently selected output language."""
    language_colors = {
        "English": "purple",
        "German": "green"
    }
    
    color = language_colors.get(desired_language, "gray")
    
    st.markdown(
        f"""
        <div style="
            padding: 5px 10px; 
            border-radius: 5px; 
            background-color: {color}25; 
            border: 1px solid {color}; 
            display: inline-block;
            margin-bottom: 10px;
            ">
            <span style="color: {color}; font-weight: bold;">
                ⚙️ Output will be in {desired_language}
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- MAIN APP FUNCTION ---
def main():
    """Main function to run the Streamlit app."""
    # Initialize the app
    client = initialize_app()
    
    # Setup sidebar and logo
    setup_sidebar()
    
    # Create and apply filters
    filtered_metadata = create_sidebar_filters()
    
    # Main content area title and language selection
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("SARA: Wiener Zeitung Archive Analysis")
    with col2:
        # Put language selection in a more prominent position
        desired_language = st.selectbox(
            "Output Language",
            ["English", "German"],
            index=0,
            format_func=lambda x: x.capitalize()
        )
    
    # Handle article selection
    article_data, selected_article_filename = handle_article_selection(filtered_metadata)
    
    # If we have an article selected, display it and feature tabs
    if article_data:
        # Display language indicator
        display_language_indicator(desired_language)
        
        # Feature tabs
        tab1, tab2 = st.tabs(["Style", "Summarization"])
        
        # Text Transformation features - pass the desired language
        with tab1:
            show_text_transformation_tab(client, article_data, selected_article_filename, desired_language)
        
        # Summarization features - pass the desired language
        with tab2:
            show_summarization_tab(client, article_data, selected_article_filename, desired_language)
            
        # Display original article
        display_article(article_data)
    
    # Display API error troubleshooting info if needed
    if st.session_state.api_error:
        show_troubleshooting_info()

if __name__ == "__main__":
    main()