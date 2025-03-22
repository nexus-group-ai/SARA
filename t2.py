import json
import os
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import tiktoken

from src.llm import get_llm_client, get_azure_embeddings_client

# Load environment variables
if not load_dotenv():
    raise Exception('Error loading .env file. Make sure to place valid keys in the .env file.')

# Constants
ARTICLES_CLEAN_DIR = os.path.join("data", "articles_clean")
FILTERED_METADATA_PATH = os.path.join("data", "filtered_metadata.csv")
ENTITY_CACHE_DIR = os.path.join("data", "entity_cache")
GRAPH_OUTPUT_DIR = os.path.join("data", "graphs")

# Ensure directories exist
os.makedirs(ENTITY_CACHE_DIR, exist_ok=True)
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

# Initialize LLM client
llm = get_llm_client(
    max_tokens=1024,
    temperature=0.1,  # Low temperature for more deterministic outputs
)

# Initialize embeddings
embeddings = get_azure_embeddings_client(
    chunk_size=512,
    show_progress_bar=True,
)

def load_filtered_articles(metadata_path):
    """Load filtered articles based on metadata."""
    metadata_df = pd.read_csv(metadata_path)
    print(f"Loaded {len(metadata_df)} articles from metadata.")
    return metadata_df

def get_article_content(filename):
    """Load article content from file."""
    file_path = os.path.join(ARTICLES_CLEAN_DIR, filename)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            article = json.load(file)
        return article
    except Exception as e:
        print(f"Error loading article {filename}: {e}")
        return None

def extract_entities(article, cache=True):
    """
    Extract entities from article using LLM.
    Returns entities in a structured format.
    """
    article_id = article.get("id")
    cache_file = os.path.join(ENTITY_CACHE_DIR, f"{article_id}.json")
    
    # Check cache first
    if cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Prepare content for LLM
    title = article.get("title", "")
    text = article.get("text", "")
    date = article.get("published_at", "")
    
    # Entity extraction prompt
    prompt = f"""
    Extract named entities from the following German news article. 
    Focus on the following entity types:
    - People (real individuals)
    - Organizations (companies, government agencies, NGOs, etc.)
    - Locations (countries, cities, specific places)
    - Events (specific happenings with dates)
    
    For each entity, provide:
    1. The entity name as it appears in the text
    2. The entity type (Person, Organization, Location, Event)
    3. A brief description based on context in the article
    4. The entity's role or significance in this article
    5. Other entities it is related to in this article
    
    Article Title: {title}
    Date: {date}
    Article Text: {text}
    
    Return ONLY a valid JSON object with the following structure:
    {{
      "entities": [
        {{
          "name": "Entity name",
          "type": "Person/Organization/Location/Event",
          "description": "Brief description",
          "significance": "Role in article",
          "related_entities": ["Related entity 1", "Related entity 2"]
        }}
      ]
    }}
    """
    
    # Call LLM
    response = llm.invoke(prompt)
    
    # Parse response
    try:
        # Extract JSON part from response if necessary
        response_text = response.strip()
        if "```json" in response_text:
            # Extract JSON part from code block
            start_idx = response_text.find("```json") + 7
            end_idx = response_text.find("```", start_idx)
            json_str = response_text[start_idx:end_idx].strip()
        elif "```" in response_text:
            # Extract from generic code block
            start_idx = response_text.find("```") + 3
            end_idx = response_text.find("```", start_idx)
            json_str = response_text[start_idx:end_idx].strip()
        else:
            # Assume the whole response is JSON
            json_str = response_text
        
        entities_data = json.loads(json_str)
        
        # Cache the results
        if cache:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entities_data, f, ensure_ascii=False, indent=2)
                
        return entities_data
    except Exception as e:
        print(f"Error parsing entity extraction response for {article_id}: {e}")
        return {"entities": []}

def build_entity_graph(entity_data_list, min_connections=1):
    """
    Build a network graph from entity data.
    
    Args:
        entity_data_list: List of entity data dictionaries
        min_connections: Minimum number of connections for an entity to be included
    
    Returns:
        NetworkX graph object
    """
    G = nx.Graph()
    
    # Track entity occurrences and connections
    entity_count = {}
    connection_count = {}
    
    # First pass: Add all entities and track occurrences
    for entity_data in entity_data_list:
        for entity in entity_data.get("entities", []):
            entity_name = entity.get("name")
            entity_type = entity.get("type")
            
            if not entity_name:
                continue
                
            # Count entity occurrences
            entity_count[entity_name] = entity_count.get(entity_name, 0) + 1
            
            # Add node if it doesn't exist
            if not G.has_node(entity_name):
                G.add_node(entity_name, 
                           type=entity_type,
                           description=entity.get("description", ""),
                           count=1)
            else:
                # Update count for existing node
                G.nodes[entity_name]["count"] = G.nodes[entity_name].get("count", 0) + 1
            
            # Track connections
            for related in entity.get("related_entities", []):
                if not related:
                    continue
                
                key = tuple(sorted([entity_name, related]))
                connection_count[key] = connection_count.get(key, 0) + 1
    
    # Second pass: Add edges based on related_entities
    for key, count in connection_count.items():
        source, target = key
        if G.has_node(source) and G.has_node(target):
            G.add_edge(source, target, weight=count)
    
    # Optional: Filter to only include entities with enough connections
    if min_connections > 0:
        nodes_to_remove = [node for node, degree in G.degree() if degree < min_connections]
        G.remove_nodes_from(nodes_to_remove)
    
    return G

def visualize_entity_graph(G, output_path=None, title="Entity Relationship Network"):
    """Visualize entity graph using matplotlib."""
    if len(G.nodes()) == 0:
        print("Graph is empty, nothing to visualize.")
        return
        
    plt.figure(figsize=(14, 10))
    
    # Define node colors by type
    color_map = {
        "Person": "skyblue",
        "Organization": "lightgreen",
        "Location": "salmon",
        "Event": "yellow"
    }
    
    # Extract node types and determine colors
    node_colors = [color_map.get(G.nodes[node].get("type"), "gray") for node in G.nodes()]
    
    # Node size based on count
    node_sizes = [30 + (G.nodes[node].get("count", 1) * 20) for node in G.nodes()]
    
    # Edge width based on weight
    edge_widths = [G[u][v].get("weight", 1) * 0.5 for u, v in G.edges()]
    
    # Layout
    pos = nx.spring_layout(G, k=0.2, seed=42)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    # Legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                  label=entity_type, markersize=10)
                      for entity_type, color in color_map.items()]
    plt.legend(handles=legend_elements, loc="upper right")
    
    plt.title(title)
    plt.axis("off")
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Graph visualization saved to {output_path}")
    
    plt.close()

def create_entity_timeline(entity_data_with_dates, entity_name):
    """
    Create a timeline visualization for a specific entity.
    
    Args:
        entity_data_with_dates: List of (entity_data, date) tuples
        entity_name: Name of the entity to create timeline for
    
    Returns:
        Timeline figure
    """
    # Filter mentions of the specified entity
    mentions = []
    for entity_data, date_str in entity_data_with_dates:
        try:
            date = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
        except:
            continue
            
        for entity in entity_data.get("entities", []):
            if entity.get("name") == entity_name:
                mentions.append((date, entity.get("significance", "")))
    
    # Sort by date
    mentions.sort(key=lambda x: x[0])
    
    if not mentions:
        print(f"No mentions found for entity {entity_name}")
        return None
    
    # Create timeline visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot points
    dates = [m[0] for m in mentions]
    y = [1] * len(dates)
    
    ax.scatter(dates, y, s=80, color='blue', zorder=2)
    ax.plot(dates, y, color='gray', linestyle='-', alpha=0.3, zorder=1)
    
    # Add annotations for significant events
    for i, (date, significance) in enumerate(mentions):
        ax.annotate(significance[:50] + "..." if len(significance) > 50 else significance,
                   (date, 1),
                   xytext=(0, 10 + (i % 3) * 10),  # Stagger annotations to avoid overlap
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   rotation=0,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Format axis
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set title and labels
    plt.title(f"Timeline for {entity_name}")
    plt.xlabel("Date")
    
    # Adjust layout
    plt.tight_layout()
    
    output_path = os.path.join(GRAPH_OUTPUT_DIR, f"timeline_{entity_name.replace(' ', '_')}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    return output_path

def main():
    # Load filtered articles
    metadata_df = load_filtered_articles(FILTERED_METADATA_PATH)
    
    # Process a subset for demo purposes
    sample_size = min(5, len(metadata_df))
    sample_df = metadata_df.sample(sample_size, random_state=42)
    
    print(f"Processing {sample_size} articles for entity extraction...")
    
    # Extract entities from each article
    entity_data_list = []
    entity_data_with_dates = []
    
    for _, row in sample_df.iterrows():
        filename = row["filename"]
        article = get_article_content(filename)
        
        if not article:
            continue
            
        print(f"Extracting entities from {filename}...")
        entity_data = extract_entities(article)
        
        if entity_data and entity_data.get("entities"):
            entity_data_list.append(entity_data)
            entity_data_with_dates.append((entity_data, article.get("published_at", "")))
    
    print(f"Entity extraction completed for {len(entity_data_list)} articles.")
    
    # Build and visualize entity graph
    print("Building entity relationship graph...")
    G = build_entity_graph(entity_data_list, min_connections=2)
    print(f"Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    
    # Visualize the graph
    graph_output = os.path.join(GRAPH_OUTPUT_DIR, "entity_network.png")
    visualize_entity_graph(G, output_path=graph_output)
    
    # Create timeline for top entities
    top_entities = sorted([(node, G.degree(node)) for node in G.nodes()], 
                         key=lambda x: x[1], reverse=True)[:5]
    
    for entity_name, degree in top_entities:
        print(f"Creating timeline for {entity_name} (connections: {degree})...")
        timeline_path = create_entity_timeline(entity_data_with_dates, entity_name)
        if timeline_path:
            print(f"Timeline saved to {timeline_path}")

if __name__ == "__main__":
    main()