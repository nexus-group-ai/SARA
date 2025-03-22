import spacy
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Set

class EntityExtractor:
    """Tool for extracting named entities from articles."""
    
    def __init__(self, model="en_core_web_lg"):
        """Initialize with specified spaCy model."""
        self.nlp = spacy.load(model)
        # Map spaCy entity types to our simplified categories
        self.entity_map = {
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION', 
            'GPE': 'PLACE',
            'LOC': 'PLACE',
            'FAC': 'PLACE',
            'NORP': 'GROUP',
            'EVENT': 'EVENT',
            'WORK_OF_ART': 'TOPIC',
            'LAW': 'TOPIC',
            'DATE': 'DATE',
            'TIME': 'DATE',
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text and categorize them.
        
        Args:
            text: The article text to process
            
        Returns:
            Dictionary mapping entity categories to lists of entities
        """
        doc = self.nlp(text)
        
        # Group entities by category
        entities = {
            'PERSON': [],
            'ORGANIZATION': [],
            'PLACE': [],
            'GROUP': [],
            'EVENT': [],
            'TOPIC': [],
            'DATE': []
        }
        
        # Collect entities
        for ent in doc.ents:
            # Map entity to our simplified categories if possible
            category = self.entity_map.get(ent.label_, 'OTHER')
            if category in entities:
                entities[category].append(ent.text)
        
        # Remove duplicates while preserving order
        for category in entities:
            seen = set()
            entities[category] = [x for x in entities[category] 
                                if not (x.lower() in seen or seen.add(x.lower()))]
        
        return entities
    
    def extract_topics(self, text: str, n=5) -> List[str]:
        """
        Extract key topics from the text based on noun phrases.
        
        Args:
            text: The article text
            n: Number of top topics to return
            
        Returns:
            List of key topics
        """
        doc = self.nlp(text)
        
        # Get noun chunks as potential topics
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                      if len(chunk.text.split()) > 1]
        
        # Count occurrences
        topic_counts = Counter(noun_phrases)
        
        # Return top N topics
        return [topic for topic, _ in topic_counts.most_common(n)]
    
    def analyze_article(self, article_text: str) -> Dict:
        """
        Perform complete analysis of an article.
        
        Args:
            article_text: The full text of the article
            
        Returns:
            Dictionary with all extracted information
        """
        # Get named entities
        entities = self.extract_entities(article_text)
        
        # Get topics
        topics = self.extract_topics(article_text)
        
        # Add topics to the results
        if 'TOPIC' in entities:
            entities['TOPIC'] = list(set(entities['TOPIC'] + topics))
        else:
            entities['TOPIC'] = topics
            
        return entities
    
    def analyze_multiple_articles(self, articles: List[str]) -> List[Dict]:
        """
        Process multiple articles and return analysis for each.
        
        Args:
            articles: List of article texts
            
        Returns:
            List of dictionaries with analysis for each article
        """
        results = []
        for article in articles:
            results.append(self.analyze_article(article))
        return results
    
    def export_to_csv(self, analysis_results: List[Dict], filepath: str) -> None:
        """
        Export analysis results to CSV file.
        
        Args:
            analysis_results: List of dictionaries with analysis results
            filepath: Path to save the CSV file
        """
        # Prepare data for DataFrame
        rows = []
        for i, result in enumerate(analysis_results):
            row = {'Article': f'Article {i+1}'}
            for category, entities in result.items():
                row[category] = ' | '.join(entities)
            rows.append(row)
        
        # Create and save DataFrame
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Sample article text
    article = """
    Apple Inc. CEO Tim Cook announced a new partnership with Microsoft 
    during a conference in San Francisco last week. The collaboration, 
    which focuses on artificial intelligence integration between iOS and 
    Windows platforms, is expected to launch in early 2025. 
    
    The announcement came as a surprise to many industry analysts, as the 
    two tech giants have historically been competitors. Sarah Johnson, 
    a senior technology analyst at Goldman Sachs, called the partnership 
    "a watershed moment for cross-platform compatibility."
    
    The event, held at the Moscone Center, also featured demonstrations of 
    upcoming Apple products, including the rumored Apple Car project.
    """
    
    # Create extractor
    extractor = EntityExtractor()
    
    # Process article
    entities = extractor.analyze_article(article)
    
    # Print results
    print("Extracted Entities:")
    for category, items in entities.items():
        if items:
            print(f"{category}: {', '.join(items)}")