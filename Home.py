import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
import openai
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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

import sys

# Add path for importing RAG modules
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Try to import RAG functionality
try:
    from rag.ask_rag import ask_question, DB_PATH
    RAG_AVAILABLE = os.path.exists(DB_PATH)
except ImportError:
    RAG_AVAILABLE = False

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
        # Initialize session state for this page
    if 'analysis_prompt' not in st.session_state:
        st.session_state.analysis_prompt = ""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = ""
    if 'entities' not in st.session_state:
        st.session_state.entities = {}
    if 'related_articles' not in st.session_state:
        st.session_state.related_articles = []
    if 'related_articles_response' not in st.session_state:
        st.session_state.related_articles_response = ""
    
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
        
        # Add demo articles checkbox
        use_demo_articles = st.checkbox("Filter to demo articles", value=False)
        
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
        
        # Filter by demo articles IDs if checked
        if use_demo_articles:
            try:
                # Load the demo article IDs from CSV
                demo_ids_df = pd.read_csv("rag/ids.csv")
                if "id" in demo_ids_df.columns:
                    # Filter metadata to only include articles with IDs in the demo list
                    filtered_metadata = metadata[metadata["id"].isin(demo_ids_df["id"])]
                    st.info(f"Filtered to {len(filtered_metadata)} demo articles")
                else:
                    st.error("Demo IDs CSV does not have 'id' column")
                    filtered_metadata = metadata.copy()
            except Exception as e:
                st.error(f"Error loading demo IDs: {e}")
                filtered_metadata = metadata.copy()
        else:
            # Apply normal date filtering if not using demo articles
            filtered_metadata = metadata[
                (metadata["published_at"].dt.date >= date_range[0]) & 
                (metadata["published_at"].dt.date <= date_range[1])
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
        
        return filtered_metadata, use_demo_articles

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

# --- ANALYSIS FUNCTIONS ---
def extract_entities(client, article_text):
    """Extract named entities from the article text using LLM."""
    prompt = f"""Extract all named entities from the following article text and classify them by type. 
    Return the results as a JSON object with the following structure:
    {{
        "people": ["Name 1", "Name 2", ...],
        "organizations": ["Org 1", "Org 2", ...],
        "locations": ["Location 1", "Location 2", ...],
        "events": ["Event 1", "Event 2", ...],
        "dates": ["Date 1", "Date 2", ...],
        "other": ["Other entity 1", "Other entity 2", ...]
    }}
    
    Article text:
    {article_text}
    
    Output only valid JSON, with no additional text before or after.
    """
    
    try:
        response = get_llm_response(client, prompt, max_tokens=1500, temperature=0.1)
        # Extract just the JSON part if there's surrounding text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            entities = json.loads(json_str)
            return entities
        else:
            st.error("Could not extract valid JSON from the model response.")
            return {}
    except Exception as e:
        st.error(f"Error extracting entities: {e}")
        return {}

def analyze_sentiment(client, article_text):
    """Analyze the sentiment and tone of the article."""
    prompt = f"""Analyze the sentiment and tone of the following article. Provide:
    1. Overall sentiment (positive, negative, neutral, or mixed)
    2. Emotional tone (e.g., optimistic, pessimistic, alarmist, hopeful, etc.)
    3. A brief explanation of your analysis
    4. A sentiment score from -1.0 (very negative) to 1.0 (very positive)
    
    Format your response as a JSON object with the following structure:
    {{
        "sentiment": "positive/negative/neutral/mixed",
        "tone": "descriptive tone",
        "explanation": "brief explanation",
        "score": 0.0
    }}
    
    Article text:
    {article_text}
    
    Output only valid JSON, with no additional text.
    """
    
    try:
        response = get_llm_response(client, prompt, max_tokens=1000, temperature=0.2)
        # Extract just the JSON part if there's surrounding text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            sentiment_data = json.loads(json_str)
            return sentiment_data
        else:
            st.error("Could not extract valid JSON from the model response.")
            return {}
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return {}

def identify_main_topics(client, article_text):
    """Identify the main topics and themes in the article."""
    prompt = f"""Identify the main topics and themes discussed in the following article.
    Return the results as a JSON object with the following structure:
    {{
        "main_topic": "The primary subject of the article",
        "subtopics": ["Subtopic 1", "Subtopic 2", ...],
        "keywords": ["Keyword 1", "Keyword 2", ...],
        "categories": ["Category 1", "Category 2", ..."],
        "summary": "A brief 1-2 sentence summary of the main point"
    }}
    
    Article text:
    {article_text}
    
    Output only valid JSON, with no additional text.
    """
    
    try:
        response = get_llm_response(client, prompt, max_tokens=1000, temperature=0.3)
        # Extract just the JSON part if there's surrounding text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            topics_data = json.loads(json_str)
            return topics_data
        else:
            st.error("Could not extract valid JSON from the model response.")
            return {}
    except Exception as e:
        st.error(f"Error identifying topics: {e}")
        return {}

def display_entity_table(entities):
    """Display entities in a table format."""
    if not entities:
        st.warning("No entities detected in the article.")
        return
    
    # Custom CSS for entity tags
    st.markdown("""
    <style>
    .entity-tag {
        display: inline-block;
        padding: 4px 8px;
        margin: 4px;
        border-radius: 4px;
        font-size: 14px;
    }
    .person { background-color: rgba(31, 119, 180, 0.2); border: 1px solid rgba(31, 119, 180, 0.8); }
    .organization { background-color: rgba(255, 127, 14, 0.2); border: 1px solid rgba(255, 127, 14, 0.8); }
    .location { background-color: rgba(44, 160, 44, 0.2); border: 1px solid rgba(44, 160, 44, 0.8); }
    .event { background-color: rgba(214, 39, 40, 0.2); border: 1px solid rgba(214, 39, 40, 0.8); }
    .date { background-color: rgba(148, 103, 189, 0.2); border: 1px solid rgba(148, 103, 189, 0.8); }
    .other { background-color: rgba(140, 86, 75, 0.2); border: 1px solid rgba(140, 86, 75, 0.8); }
    </style>
    """, unsafe_allow_html=True)
    
    # Create columns for each entity type
    cols = st.columns(3)
    
    entity_labels = {
        'people': 'People', 
        'organizations': 'Organizations', 
        'locations': 'Locations',
        'events': 'Events', 
        'dates': 'Dates', 
        'other': 'Other'
    }
    
    entity_classes = {
        'people': 'person', 
        'organizations': 'organization', 
        'locations': 'location',
        'events': 'event', 
        'dates': 'date', 
        'other': 'other'
    }
    
    # Display entities in columns
    for i, (entity_type, label) in enumerate(entity_labels.items()):
        col_idx = i % 3
        with cols[col_idx]:
            st.subheader(label)
            if entity_type in entities and entities[entity_type]:
                entity_html = ""
                for entity in entities[entity_type]:
                    css_class = entity_classes.get(entity_type, 'other')
                    entity_html += f'<span class="entity-tag {css_class}">{entity}</span> '
                st.markdown(entity_html, unsafe_allow_html=True)
            else:
                st.write("None detected")

def display_sentiment_gauge(sentiment_data):
    """Display a gauge chart for sentiment score."""
    if not sentiment_data or 'score' not in sentiment_data:
        st.warning("No sentiment data available.")
        return
    
    score = sentiment_data.get('score', 0)
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Sentiment Score"},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.6], 'color': "firebrick"},
                {'range': [-0.6, -0.2], 'color': "salmon"},
                {'range': [-0.2, 0.2], 'color': "lightgray"},
                {'range': [0.2, 0.6], 'color': "lightgreen"},
                {'range': [0.6, 1], 'color': "forestgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display sentiment details
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Overall Sentiment**: {sentiment_data.get('sentiment', 'N/A')}")
    with col2:
        st.info(f"**Emotional Tone**: {sentiment_data.get('tone', 'N/A')}")
    
    st.markdown(f"**Analysis**: {sentiment_data.get('explanation', 'No explanation provided.')}")

def display_topic_analysis(topics_data):
    """Display topic analysis results."""
    if not topics_data or 'main_topic' not in topics_data:
        st.warning("No topic data available.")
        return
    
    st.subheader("Topic Analysis")
    st.markdown(f"**Main Topic**: {topics_data.get('main_topic', 'N/A')}")
    st.markdown(f"**Summary**: {topics_data.get('summary', 'No summary available.')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Subtopics**")
        subtopics = topics_data.get('subtopics', [])
        if subtopics:
            for topic in subtopics:
                st.markdown(f"- {topic}")
        else:
            st.write("No subtopics identified")
    
    with col2:
        st.markdown("**Keywords**")
        keywords = topics_data.get('keywords', [])
        if keywords:
            # Display as pills/tags
            keyword_html = ""
            for keyword in keywords:
                keyword_html += f'<span style="background-color: #e0f7fa; padding: 3px 8px; margin: 2px; border-radius: 12px; font-size: 14px; display: inline-block;">{keyword}</span> '
            st.markdown(keyword_html, unsafe_allow_html=True)
        else:
            st.write("No keywords identified")
    
    # Categories as a horizontal bar chart if available
    categories = topics_data.get('categories', [])
    if categories:
        st.markdown("**Categories**")
        # Create a simple horizontal chart with equal values
        df_cat = pd.DataFrame({
            'category': categories,
            'value': [1] * len(categories)  # Equal weight for visualization
        })
        
        fig = px.bar(
            df_cat, 
            y='category', 
            x='value',
            orientation='h',
            labels={'category': '', 'value': ''},
            height=min(100 + len(categories) * 30, 400)
        )
        fig.update_layout(showlegend=False)
        fig.update_traces(marker_color='skyblue')
        fig.update_xaxes(showticklabels=False, showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)

def show_entities_tab(client, article_data):
    st.subheader("Entity Extraction")
    st.write("Identify and classify named entities mentioned in the article.")
    
    if st.button("Extract Entities", key="extract_entities_btn"):
        with st.spinner("Extracting entities..."):
            entities = extract_entities(client, article_data["text"])
            st.session_state.entities = entities
    
    # Display entities if available
    if st.session_state.entities:
        display_entity_table(st.session_state.entities)

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

def find_related_articles(article_text):
    """Find related articles using the RAG pipeline."""
    
    if not RAG_AVAILABLE:
        return "RAG functionality is not available. Please check if the vector database exists.", []
    
    query = article_text[:500]  # Use the first 500 chars
    
    try:
        # Call the RAG pipeline
        response = ask_question(query)
        
        # Extract source documents information
        sources = []
        if "source_documents" in response:
            for doc in response["source_documents"]:
                sources.append(doc.metadata)
        
        return response["result"], sources
    except Exception as e:
        st.error(f"Error finding related articles: {e}")
        return f"An error occurred: {str(e)}", []

# New related tab UI function
def show_related_tab(client, article_data, metadata_df, desired_language):
    """Display the related articles tab with RAG functionality."""
    st.subheader("Related Articles")
    
    if not RAG_AVAILABLE:
        st.warning("RAG functionality is not available. Please check if the vector database exists at 'rag/ai_topic.db'.")
        st.info("To enable this feature, first run the 'rag/create_rag.py' script to build the vector database.")
        return
    
    # Execute search
    if st.button("Find Related Articles", key="find_related_button"):
        with st.spinner("Searching for related articles..."):
            try:
                response, sources = find_related_articles(
                    article_data["text"]
                )
                
                # Save results to session state
                st.session_state.related_articles_response = response
                st.session_state.related_articles = sources
                
                # Display response
                st.markdown("### Search Results")
                
                # Display source documents
                if sources:
                    st.markdown("### Source Articles")
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"{i}. {source.get('title', 'Untitled')} - {source.get('published_at', 'Unknown date')}"):
                            st.markdown(f"**Author:** {source.get('author', 'Unknown')}")
                            st.markdown(f"**Published:** {source.get('published_at', 'Unknown')}")
                            st.markdown(f"**ID:** {source.get('id', 'Unknown')}")
                            
                            # Add button to load the full article
                            if st.button(f"Load full article", key=f"load_source_{i}"):
                                source_id = source.get('id')
                                if source_id and not metadata_df.empty:
                                    matching_rows = metadata_df[metadata_df['id'] == source_id]
                                    if not matching_rows.empty:
                                        filename = matching_rows.iloc[0]['filename']
                                        source_article = load_article(filename)
                                        if source_article:
                                            st.subheader(source_article["title"])
                                            st.markdown(source_article["text"].replace("\n", "\n\n"))
                                        else:
                                            st.error("Could not load the article.")
                                    else:
                                        st.error("Article not found in metadata.")
                                else:
                                    st.error("Article ID not available or metadata not loaded.")
                else:
                    st.info("No source articles found.")
                
            except Exception as e:
                st.error(f"Error finding related articles: {e}")
                st.session_state.api_error = True


# --- MAIN APP FUNCTION ---
def main():
    """Main function to run the Streamlit app."""
    # Initialize the app
    client = initialize_app()
    
    # Setup sidebar and logo
    setup_sidebar()
    
    # Create and apply filters
    filtered_metadata, use_demo_articles = create_sidebar_filters()
    
    # Main content area title and language selection
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("SARA: Wiener Zeitung Archive")
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
        tabs = st.tabs(["Style", "Summarize", "Entities", "Sentiment", "Topic", "Related"])
        
        # Text Transformation features - pass the desired language
        with tabs[0]:
            show_text_transformation_tab(client, article_data, selected_article_filename, desired_language)
        
        # Summarization features - pass the desired language
        with tabs[1]:
            show_summarization_tab(client, article_data, selected_article_filename, desired_language)
            
        # Entity Extraction features
        with tabs[2]:
            show_entities_tab(client, article_data)
        
        # Sentiment Analysis features
        with tabs[3]:
            st.subheader("Sentiment Analysis")
            st.write("Analyze the sentiment and tone of the article.")
            
            if st.button("Analyze Sentiment", key="analyze_sentiment_btn"):
                with st.spinner("Analyzing sentiment..."):
                    sentiment_data = analyze_sentiment(client, article_data["text"])
                    st.session_state.sentiment_data = sentiment_data
                
                # Display sentiment analysis if available
                if st.session_state.sentiment_data:
                    display_sentiment_gauge(st.session_state.sentiment_data)
        
        # Topic Identification features
        with tabs[4]:
            st.subheader("Topic Identification")
            st.write("Identify the main topics and themes in the article.")
            
            if st.button("Identify Topics", key="identify_topics_btn"):
                with st.spinner("Identifying topics..."):
                    topics_data = identify_main_topics(client, article_data["text"])
                    st.session_state.topics_data = topics_data
                
                # Display topic analysis if available
                if st.session_state.topics_data:
                    display_topic_analysis(st.session_state.topics_data)
        
        with tabs[5]:
            if use_demo_articles:
                show_related_tab(client, article_data, load_metadata(), desired_language)
            else:
                st.error("Only available if Filter to Demo Articles is selected")
            
            
        # Display original article
        display_article(article_data)
    
    # Display API error troubleshooting info if needed
    if st.session_state.api_error:
        show_troubleshooting_info()

if __name__ == "__main__":
    main()