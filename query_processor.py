# query_processor.py
import re
from typing import List, Dict, Tuple
import numpy as np

class QueryProcessor:
    def __init__(self, embedding_generator, search_database):
        self.embedding_generator = embedding_generator
        self.search_database = search_database
    
    def parse_complex_query(self, query: str) -> Dict:
        """Parse any type of query for AND/OR logic"""
        query_lower = query.lower()
        
        # Split by 'and' to get components
        and_components = [comp.strip() for comp in re.split(r'\band\b', query_lower)]
        
        # Split by 'or' to get alternate components
        or_components = []
        for component in and_components:
            or_parts = [part.strip() for part in re.split(r'\bor\b', component)]
            or_components.extend(or_parts)
        
        parsed = {
            'components': and_components,
            'or_components': or_components,
            'full_query': query,
            'logic': 'AND' if len(and_components) > 1 else ('OR' if len(or_components) > 1 else 'SINGLE')
        }
        
        return parsed
    
    def search_segments(self, query: str, k: int = 20) -> List[Tuple[Dict, float]]:
        """Search for segments matching any type of query"""
        try:
            # Use the specialized query embedding method
            query_embedding = self.embedding_generator.generate_query_embedding_for_search(query)
            
            # Search database
            results = self.search_database.search(query_embedding, k)
            return results
            
        except Exception as e:
            print(f"Error in search_segments: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def filter_by_logic(self, query: str, results: List[Tuple[Dict, float]], 
                       min_score: float = 0.3) -> List[Tuple[Dict, float]]:
        """Filter results based on query logic and minimum score"""
        if not results:
            return []
        
        # Filter by minimum score first
        scored_results = [(segment, score) for segment, score in results if score >= min_score]
        
        parsed_query = self.parse_complex_query(query)
        
        if parsed_query['logic'] == 'SINGLE':
            return scored_results
        
        # For complex queries, apply semantic filtering
        filtered_results = []
        
        for segment_data, score in scored_results:
            segment_text = segment_data.get('transcript', '').lower()
            
            if parsed_query['logic'] == 'AND':
                # For AND logic, check if multiple components are present
                component_matches = 0
                for component in parsed_query['components']:
                    # Use semantic similarity instead of exact word matching
                    component_words = component.split()
                    if any(word in segment_text for word in component_words):
                        component_matches += 1
                
                # Include if matches multiple components OR has very high similarity
                if component_matches >= len(parsed_query['components']) * 0.5 or score > 0.7:
                    filtered_results.append((segment_data, score))
            
            elif parsed_query['logic'] == 'OR':
                # For OR logic, any component match is sufficient
                for component in parsed_query['or_components']:
                    component_words = component.split()
                    if any(word in segment_text for word in component_words) or score > 0.6:
                        filtered_results.append((segment_data, score))
                        break
        
        return filtered_results
    
    def enhance_search_with_context(self, query: str, k: int = 20, 
                                  context_window: float = 10.0) -> List[Tuple[Dict, float]]:
        """Enhanced search that considers temporal context"""
        try:
            # Get initial search results
            results = self.search_segments(query, k * 2)  # Get more results initially
            
            if not results:
                return []
            
            # Group results by video and timestamp
            video_groups = {}
            for segment, score in results:
                video_id = segment['video_id']
                timestamp = segment['timestamp']
                
                if video_id not in video_groups:
                    video_groups[video_id] = []
                video_groups[video_id].append((segment, score, timestamp))
            
            # Enhanced results considering context
            enhanced_results = []
            
            for video_id, segments in video_groups.items():
                # Sort by timestamp
                segments.sort(key=lambda x: x[2])
                
                # Find clusters of related segments
                for segment, score, timestamp in segments:
                    # Check for nearby high-scoring segments
                    context_boost = 0.0
                    nearby_segments = [
                        s for s in segments 
                        if abs(s[2] - timestamp) <= context_window and s[1] > 0.4
                    ]
                    
                    if len(nearby_segments) > 1:
                        context_boost = 0.1 * (len(nearby_segments) - 1)
                    
                    enhanced_score = min(score + context_boost, 1.0)
                    enhanced_results.append((segment, enhanced_score))
            
            # Sort by enhanced score and return top k
            enhanced_results.sort(key=lambda x: x[1], reverse=True)
            return enhanced_results[:k]
            
        except Exception as e:
            print(f"Error in enhanced search: {e}")
            return self.search_segments(query, k)
