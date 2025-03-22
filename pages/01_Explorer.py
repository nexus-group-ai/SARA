import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Explorer",
    page_icon="📰",
    layout="wide"
)

st.header("Explorer")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/metadata.csv')
        # Convert published_at to datetime format
        df['published_at'] = pd.to_datetime(df['published_at'])
        # Extract year and month
        df['year'] = df['published_at'].dt.year
        df['month'] = df['published_at'].dt.month
        df['year_month'] = df['published_at'].dt.strftime('%Y-%m')
        # Add decade
        df['decade'] = (df['year'] // 10) * 10
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Timeline Analysis", "Content Categories", "Topics & Tags", "Dataset Explorer"])
    
    with tab1:
        # Overview section with key metrics
        st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Total Articles", f"{df.shape[0]:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Date Range", f"{df['published_at'].min().strftime('%b %Y')} - {df['published_at'].max().strftime('%b %Y')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Unique Authors", f"{df['author'].nunique():,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            avg_words = int(df['word_count'].mean())
            st.metric("Avg. Word Count", f"{avg_words:,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Word count distribution
        st.markdown('<div class="sub-header">Word Count Distribution</div>', unsafe_allow_html=True)
        
        fig_word_count = px.histogram(
            df, 
            x='word_count', 
            nbins=50,
            labels={'word_count': 'Word Count', 'count': 'Number of Articles'},
            title='Distribution of Article Lengths'
        )
        fig_word_count.update_layout(
            xaxis_title="Word Count",
            yaxis_title="Number of Articles",
            bargap=0.1
        )
        st.plotly_chart(fig_word_count, use_container_width=True)
        
        # Publication trend by year
        st.markdown('<div class="sub-header">Publication Trend</div>', unsafe_allow_html=True)
        
        yearly_counts = df.groupby('year').size().reset_index(name='count')
        fig_yearly = px.line(
            yearly_counts, 
            x='year', 
            y='count',
            labels={'year': 'Year', 'count': 'Number of Articles'},
            title='Articles Published by Year'
        )
        fig_yearly.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Articles"
        )
        st.plotly_chart(fig_yearly, use_container_width=True)
        
    with tab2:
        st.markdown('<div class="sub-header">Timeline Analysis</div>', unsafe_allow_html=True)
        
        # Year range selector
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        year_range = st.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        # Monthly publication trend
        monthly_counts = filtered_df.groupby('year_month').size().reset_index(name='count')
        monthly_counts['year_month'] = pd.to_datetime(monthly_counts['year_month'])
        monthly_counts = monthly_counts.sort_values('year_month')
        
        fig_monthly = px.line(
            monthly_counts, 
            x='year_month', 
            y='count',
            labels={'year_month': 'Month', 'count': 'Number of Articles'},
            title='Monthly Publication Trend'
        )
        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Articles"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Section volume by year
        section_by_year = filtered_df.groupby(['year', 'section']).size().reset_index(name='count')
        fig_section_year = px.area(
            section_by_year, 
            x='year', 
            y='count', 
            color='section',
            labels={'year': 'Year', 'count': 'Number of Articles', 'section': 'Section'},
            title='Publication Volume by Section Over Time'
        )
        fig_section_year.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Articles"
        )
        st.plotly_chart(fig_section_year, use_container_width=True)
        
        # Word count trend over time
        word_count_year = filtered_df.groupby('year')['word_count'].mean().reset_index()
        fig_word_trend = px.line(
            word_count_year, 
            x='year', 
            y='word_count',
            labels={'year': 'Year', 'word_count': 'Average Word Count'},
            title='Average Article Length Over Time'
        )
        fig_word_trend.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Word Count"
        )
        st.plotly_chart(fig_word_trend, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">Content Categories</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Section distribution
            section_counts = df['section'].value_counts().reset_index()
            section_counts.columns = ['section', 'count']
            
            fig_section = px.pie(
                section_counts, 
                values='count', 
                names='section',
                title='Distribution of Articles by Section'
            )
            fig_section.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_section, use_container_width=True)
        
        with col2:
            # Top categories
            top_categories = df['category'].value_counts().head(10).reset_index()
            top_categories.columns = ['category', 'count']
            
            fig_category = px.bar(
                top_categories, 
                y='category', 
                x='count',
                orientation='h',
                title='Top 10 Categories',
                labels={'category': 'Category', 'count': 'Number of Articles'}
            )
            fig_category.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_category, use_container_width=True)
            
        # Word count by section
        section_words = df.groupby('section')['word_count'].mean().reset_index()
        fig_section_words = px.bar(
            section_words, 
            x='section', 
            y='word_count',
            title='Average Word Count by Section',
            labels={'section': 'Section', 'word_count': 'Average Word Count'}
        )
        st.plotly_chart(fig_section_words, use_container_width=True)
        
        # Top authors
        if df['author'].isna().mean() < 0.5:  # Only show if at least 50% of articles have author info
            top_authors = df[df['author'].notna()]['author'].value_counts().head(15).reset_index()
            top_authors.columns = ['author', 'count']
            
            fig_authors = px.bar(
                top_authors, 
                y='author', 
                x='count',
                orientation='h',
                title='Top 15 Authors by Article Count',
                labels={'author': 'Author', 'count': 'Number of Articles'}
            )
            fig_authors.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_authors, use_container_width=True)
    
    with tab4:
        st.markdown('<div class="sub-header">Topics & Tags</div>', unsafe_allow_html=True)
        st.info("Note: Tags were assigned with a zero-shot model and might be inaccurate. The 'valid_indicator' hints at more trustworthy tags (true for ~10% of data).")
        
        # Tag selector
        tag_options = ['financial_crisis', 'sustainability', 'fake_news', 'ai', 'digitalization', 
                      'local_journalism', 'covid', 'demographics', 'innovation']
        selected_tag = st.selectbox("Select Tag to Analyze", tag_options)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tag distribution
            tag_threshold = st.slider(f"Threshold for {selected_tag}", 0.0, 1.0, 0.7)
            df['tag_present'] = df[selected_tag] > tag_threshold
            
            tag_present_count = df['tag_present'].value_counts().reset_index()
            tag_present_count.columns = ['Present', 'Count']
            tag_present_count['Present'] = tag_present_count['Present'].map({True: f"{selected_tag} > {tag_threshold}", False: f"{selected_tag} ≤ {tag_threshold}"})
            
            fig_tag = px.pie(
                tag_present_count, 
                values='Count', 
                names='Present',
                title=f'Articles with {selected_tag} Score > {tag_threshold}'
            )
            fig_tag.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_tag, use_container_width=True)
        
        with col2:
            # Tag score distribution
            fig_tag_dist = px.histogram(
                df, 
                x=selected_tag,
                nbins=50,
                title=f'Distribution of {selected_tag} Scores',
                labels={selected_tag: 'Score', 'count': 'Number of Articles'}
            )
            st.plotly_chart(fig_tag_dist, use_container_width=True)
        
        # Tag score over time
        tag_by_year = df.groupby('year')[selected_tag].mean().reset_index()
        
        fig_tag_trend = px.line(
            tag_by_year, 
            x='year', 
            y=selected_tag,
            labels={'year': 'Year', selected_tag: f'Average {selected_tag} Score'},
            title=f'Average {selected_tag} Score Over Time'
        )
        st.plotly_chart(fig_tag_trend, use_container_width=True)
        
        # Correlation matrix of tags
        st.markdown('<div class="sub-header">Tag Correlations</div>', unsafe_allow_html=True)
        
        corr_matrix = df[tag_options].corr()
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(x="Tag", y="Tag", color="Correlation"),
            x=tag_options,
            y=tag_options,
            title="Correlation Between Tags"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Top articles by selected tag
        st.markdown(f'<div class="sub-header">Top Articles by {selected_tag} Score</div>', unsafe_allow_html=True)
        
        top_by_tag = df.sort_values(by=selected_tag, ascending=False).head(10)[['title', 'published_at', 'author', 'category', selected_tag]]
        top_by_tag.columns = ['Title', 'Published Date', 'Author', 'Category', f'{selected_tag} Score']
        top_by_tag['Published Date'] = top_by_tag['Published Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(top_by_tag, use_container_width=True)
        
    with tab5:
        st.markdown('<div class="sub-header">Dataset Explorer</div>', unsafe_allow_html=True)
        st.write("Use the filters below to explore and create a subset of the dataset for your hackathon project.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Date range filter
            date_range = st.date_input(
                "Select Date Range",
                value=[df['published_at'].min().date(), df['published_at'].max().date()],
                min_value=df['published_at'].min().date(),
                max_value=df['published_at'].max().date()
            )
            
            # Convert to datetime for filtering
            if len(date_range) == 2:
                start_date, end_date = date_range
                start_date = datetime.combine(start_date, datetime.min.time())
                end_date = datetime.combine(end_date, datetime.max.time())
            else:
                start_date = datetime.combine(date_range[0], datetime.min.time())
                end_date = datetime.combine(date_range[0], datetime.max.time())
        
        with col2:
            # Section filter
            sections = ['All'] + sorted(df['section'].unique().tolist())
            selected_section = st.selectbox("Select Section", sections)
            
            # Category filter (dependent on section)
            if selected_section != 'All':
                categories = ['All'] + sorted(df[df['section'] == selected_section]['category'].unique().tolist())
            else:
                categories = ['All'] + sorted(df['category'].unique().tolist())
            
            selected_category = st.selectbox("Select Category", categories)
        
        with col3:
            # Tag filter
            selected_filter_tag = st.selectbox("Filter by Tag Score", ['None'] + tag_options)
            
            if selected_filter_tag != 'None':
                tag_filter_value = st.slider(f"{selected_filter_tag} Score Threshold", 0.0, 1.0, 0.7)
            
            # Word count filter
            word_count_range = st.slider(
                "Word Count Range", 
                int(df['word_count'].min()), 
                int(df['word_count'].max()), 
                (int(df['word_count'].min()), int(df['word_count'].max()))
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        # Date filter
        filtered_df = filtered_df[(filtered_df['published_at'] >= start_date) & 
                                 (filtered_df['published_at'] <= end_date)]
        
        # Section filter
        if selected_section != 'All':
            filtered_df = filtered_df[filtered_df['section'] == selected_section]
        
        # Category filter
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        # Tag filter
        if selected_filter_tag != 'None':
            filtered_df = filtered_df[filtered_df[selected_filter_tag] > tag_filter_value]
        
        # Word count filter
        filtered_df = filtered_df[(filtered_df['word_count'] >= word_count_range[0]) & 
                                (filtered_df['word_count'] <= word_count_range[1])]
        
        # Display filtered dataset info
        st.markdown(f"**Filtered Dataset: {filtered_df.shape[0]:,} articles**")
        
        # Display sample of filtered data
        st.dataframe(filtered_df[['title', 'published_at', 'author', 'category', 'section', 'word_count']].head(100), use_container_width=True)
        
        
        # Statistics
        st.markdown('<div class="sub-header">Filtered Dataset Statistics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not filtered_df.empty:
                # Publication trend of filtered data
                yearly_filtered = filtered_df.groupby('year').size().reset_index(name='count')
                
                fig_yearly_filtered = px.line(
                    yearly_filtered, 
                    x='year', 
                    y='count',
                    labels={'year': 'Year', 'count': 'Number of Articles'},
                    title='Filtered Articles by Year'
                )
                st.plotly_chart(fig_yearly_filtered, use_container_width=True)
            else:
                st.warning("No data matches the selected filters.")
                
        with col2:
            if not filtered_df.empty and 'section' in filtered_df.columns:
                # Section distribution in filtered data
                section_filtered = filtered_df['section'].value_counts().reset_index()
                section_filtered.columns = ['section', 'count']
                
                fig_section_filtered = px.pie(
                    section_filtered, 
                    values='count', 
                    names='section',
                    title='Sections in Filtered Dataset'
                )
                fig_section_filtered.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_section_filtered, use_container_width=True)

else:
    st.error("Failed to load data. Please check if 'data/metadata.csv' exists and is valid.")