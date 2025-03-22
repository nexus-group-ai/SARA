import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import openai
from datetime import datetime
import numpy as np

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
    if 'entities' not in st.session_state:
        st.session_state.entities = {}
    
    return client

# --- SIDEBAR SETUP ---
def setup_sidebar():
    """Set up the sidebar UI components."""
    with st.sidebar:
        st.logo(image=IMG_PATH_LOGO_FULL, link="https://nexus-group.ai", icon_image=IMG_PATH_LOGO_ICON, size="large")
        st.title("SARA")
        st.caption("Summarize. Analyze. Retrieve. Annotate.")
        st.markdown("---")

# --- DATA HANDLING FUNCTIONS ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_metadata():
    """Load and preprocess the metadata CSV file."""
    try:
        df = pd.read_csv(METADATA_PATH)
        # Convert date columns immediately to avoid repeated conversions
        df["published_at"] = pd.to_datetime(df["published_at"])
        # Add derived columns
        df['year'] = df['published_at'].dt.year
        df['month'] = df['published_at'].dt.month
        df['year_month'] = df['published_at'].dt.strftime('%Y-%m')
        df['decade'] = (df['year'] // 10) * 10
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
        st.session_state.analysis_prompt = prompt
        
        system_prompt = "You are SARA, an intelligent assistant for news article analysis specialized in entity extraction and content analysis."
        
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
        if hasattr(e, 'response'):
            st.error(f"Response details: {e.response}")
        return "Sorry, I encountered an error while processing your request. Please check the API configuration."

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

def sidebar_filters():
    """Create filter controls in the sidebar."""
    with st.sidebar:
        st.subheader("Filter Articles")
        
        # Load metadata for filtering
        metadata = load_metadata()
        if metadata.empty:
            st.error("No metadata available")
            return pd.DataFrame()
        
        # Date range filter
        date_range = st.date_input(
            "Date range",
            [datetime(2010, 1, 1), datetime(2023, 6, 30)],
            format="YYYY-MM-DD",
        )
        
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

def display_entity_network(entities):
    """Display a network visualization of entities."""
    if not entities or sum(len(ents) for ents in entities.values()) == 0:
        st.warning("No entities found to visualize.")
        return
    
    # Setup nodes and links data
    nodes = []
    node_types = {
        'people': 'Person', 
        'organizations': 'Organization', 
        'locations': 'Location',
        'events': 'Event',
        'dates': 'Date',
        'other': 'Other'
    }
    
    # Color mapping for entity types
    colors = {
        'Person': 'rgba(31, 119, 180, 0.8)',
        'Organization': 'rgba(255, 127, 14, 0.8)',
        'Location': 'rgba(44, 160, 44, 0.8)',
        'Event': 'rgba(214, 39, 40, 0.8)',
        'Date': 'rgba(148, 103, 189, 0.8)',
        'Other': 'rgba(140, 86, 75, 0.8)'
    }
    
    # Create nodes
    for entity_type, entity_list in entities.items():
        if entity_type in node_types and isinstance(entity_list, list):
            for entity in entity_list:
                if entity and isinstance(entity, str):
                    nodes.append({
                        'id': entity,
                        'label': entity,
                        'type': node_types.get(entity_type, 'Other'),
                        'color': colors.get(node_types.get(entity_type, 'Other'))
                    })
    
    # Create a simple force layout with Plotly
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    # Generate positions randomly in a circle
    for i, node in enumerate(nodes):
        theta = 2 * np.pi * i / len(nodes)
        radius = 10
        node_x.append(radius * np.cos(theta))
        node_y.append(radius * np.sin(theta))
        node_text.append(f"{node['label']} ({node['type']})")
        node_color.append(node['color'])
    
    # Create a scatter plot for nodes
    fig = go.Figure()
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=15,
            color=node_color,
            line=dict(width=1, color='white')
        ),
        text=node_text,
        textposition="top center",
        hoverinfo='text'
    ))
    
    # Update layout
    fig.update_layout(
        title="Entity Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

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

# --- CUSTOM CSS ---
def add_custom_css():
    """Add custom CSS styles."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
        margin-bottom: 1rem !important;
    }
    .sub-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #2563EB !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    .card {
        border-radius: 5px !important;
        padding: 1.5rem !important;
        background-color: #F8FAFC !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24) !important;
        margin-bottom: 1rem !important;
    }
    .analysis-tab {
        padding: 20px !important;
        border-radius: 5px !important;
        margin-top: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MAIN PAGE FUNCTIONALITY ---
def main():
    """Main function to run the Article Analysis page."""
    # Initialize the page
    client = initialize_page()
    
    # Add custom CSS
    add_custom_css()
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content
    st.markdown('<div class="main-header">Article Analysis</div>', unsafe_allow_html=True)
    st.markdown("Extract insights and analyze content from Wiener Zeitung articles using AI.")
    
    # Get filtered metadata
    filtered_metadata = sidebar_filters()
    
    # Check if we have articles
    if filtered_metadata.empty:
        st.warning("No articles match your current filters. Please adjust your criteria.")
        return
    
    # Article selection
    st.subheader("Select an Article to Analyze")
    
    if len(filtered_metadata) > 1000:
        st.warning(f"Too many articles ({len(filtered_metadata)}). Please narrow your selection with more filters.")
        return
    
    # Display article selection dropdown
    selected_article_idx = st.selectbox(
        f"Choose from {len(filtered_metadata)} articles:", 
        range(len(filtered_metadata)),
        format_func=lambda x: f"{filtered_metadata.iloc[x]['published_date_str']} - {filtered_metadata.iloc[x]['title']}"
    )
    
    selected_article_filename = filtered_metadata.iloc[selected_article_idx]["filename"]
    article_data = load_article(selected_article_filename)
    
    if not article_data:
        st.error("Could not load the selected article.")
        return
    
    # Display article header
    st.markdown('<div class="sub-header">Selected Article</div>', unsafe_allow_html=True)
    st.markdown(f"**{article_data['title']}**")
    st.caption(f"Author: {article_data['author']} | Published: {article_data['published_at']}")
    st.caption(f"Category: {article_data['category']} | Section: {article_data['section']}")
    
    # Create analysis tabs
    tabs = st.tabs(["Entity Extraction", "Sentiment Analysis", "Topic Analysis", "Full Text"])
    
    # Entity Extraction tab
    with tabs[0]:
        st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
        st.subheader("Entity Extraction")
        st.write("Identify and classify named entities mentioned in the article.")
        
        if st.button("Extract Entities", key="extract_entities_btn"):
            with st.spinner("Extracting entities..."):
                entities = extract_entities(client, article_data["text"])
                st.session_state.entities = entities
        
        # Display entities if available
        if st.session_state.entities:
            # Display both visualizations
            entity_view = st.radio(
                "Entity Display",
                ["Table View", "Network Visualization"],
                horizontal=True
            )
            
            if entity_view == "Table View":
                display_entity_table(st.session_state.entities)
            else:
                display_entity_network(st.session_state.entities)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sentiment Analysis tab
    with tabs[1]:
        st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
        st.subheader("Sentiment Analysis")
        st.write("Analyze the overall sentiment, tone, and emotional content of the article.")
        
        if st.button("Analyze Sentiment", key="analyze_sentiment_btn"):
            with st.spinner("Analyzing sentiment..."):
                sentiment_data = analyze_sentiment(client, article_data["text"])
                if sentiment_data:
                    display_sentiment_gauge(sentiment_data)
                else:
                    st.error("Failed to analyze sentiment.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Topic Analysis tab
    with tabs[2]:
        st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
        st.subheader("Topic Analysis")
        st.write("Identify main topics, themes, and keywords in the article.")
        
        if st.button("Identify Topics", key="identify_topics_btn"):
            with st.spinner("Analyzing topics..."):
                topics_data = identify_main_topics(client, article_data["text"])
                if topics_data:
                    display_topic_analysis(topics_data)
                else:
                    st.error("Failed to identify topics.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Full Text tab
    with tabs[3]:
        st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
        st.subheader("Full Article Text")
        st.markdown(article_data["text"].replace("\n", "\n\n"))
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()