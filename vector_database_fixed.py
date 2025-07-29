# vector_database_fixed.py (complete version with get_statistics)
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import os

class VideoSearchDatabase:
    def __init__(self, embedding_dim: int = 896):
        """Initialize vector database with sklearn backend"""
        self.embedding_dim = embedding_dim
        self.embeddings = []
        self.metadata = []
        
    def add_segments(self, embeddings: List[np.ndarray], metadata: List[Dict]):
        """Add video segments to the database"""
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")
        
        if not embeddings:
            print("Warning: No embeddings provided to add_segments")
            return
        
        try:
            # Convert to numpy array and validate dimensions
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Check embedding dimensions
            if embeddings_array.shape[1] != self.embedding_dim:
                print(f"Warning: Embedding dimension mismatch. Expected {self.embedding_dim}, got {embeddings_array.shape[1]}")
                # Adjust embedding dimension if needed
                if embeddings_array.shape[1] < self.embedding_dim:
                    # Pad with zeros
                    padding = np.zeros((embeddings_array.shape[0], self.embedding_dim - embeddings_array.shape[1]), dtype=np.float32)
                    embeddings_array = np.concatenate([embeddings_array, padding], axis=1)
                else:
                    # Truncate
                    embeddings_array = embeddings_array[:, :self.embedding_dim]
            
            # L2 normalize embeddings
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / (norms + 1e-8)
            
            # Add to database
            self.embeddings.extend(embeddings_array.tolist())
            self.metadata.extend(metadata)
            
            print(f"Added {len(embeddings)} segments to database. Total segments: {len(self.metadata)}")
            
        except Exception as e:
            print(f"Error adding segments to database: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Dict, float]]:
        """Search for similar segments using cosine similarity"""
        if not self.embeddings:
            print("Warning: No embeddings in database for search")
            return []
        
        try:
            # Validate and normalize query
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Handle dimension mismatch
            if len(query_embedding) != self.embedding_dim:
                if len(query_embedding) < self.embedding_dim:
                    # Pad with zeros
                    padding = np.zeros(self.embedding_dim - len(query_embedding), dtype=np.float32)
                    query_embedding = np.concatenate([query_embedding, padding])
                else:
                    # Truncate
                    query_embedding = query_embedding[:self.embedding_dim]
            
            query_embedding = query_embedding.reshape(1, -1)
            query_norm = np.linalg.norm(query_embedding)
            
            if query_norm == 0:
                print("Warning: Query embedding has zero norm")
                return []
            
            query_embedding = query_embedding / query_norm
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, embeddings_array)[0]
            
            # Get top k results
            k = min(k, len(similarities))  # Ensure k doesn't exceed available results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.metadata) and similarities[idx] > 0:  # Only include positive similarities
                    results.append((self.metadata[idx], float(similarities[idx])))
            
            return results
        
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {
            'total_segments': len(self.metadata),
            'embedding_dimension': self.embedding_dim,
            'videos': {}
        }
        
        # Count segments per video
        for segment in self.metadata:
            video_id = segment.get('video_id', 'unknown')
            if video_id not in stats['videos']:
                stats['videos'][video_id] = 0
            stats['videos'][video_id] += 1
        
        stats['total_videos'] = len(stats['videos'])
        return stats
    
    def remove_video(self, video_id: str) -> int:
        """Remove all segments from a specific video"""
        indices_to_remove = []
        for i, metadata in enumerate(self.metadata):
            if metadata.get('video_id') == video_id:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for idx in reversed(indices_to_remove):
            del self.embeddings[idx]
            del self.metadata[idx]
        
        print(f"Removed {len(indices_to_remove)} segments for video: {video_id}")
        return len(indices_to_remove)
    
    def clear(self):
        """Clear all data from the database"""
        self.embeddings = []
        self.metadata = []
        print("Database cleared")
    
    def save(self, filepath: str):
        """Save embeddings and metadata to file"""
        try:
            data = {
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'version': '1.0'  # For future compatibility
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Database saved to {filepath}.pkl with {len(self.metadata)} segments")
            
        except Exception as e:
            print(f"Error saving database: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """Load embeddings and metadata from file"""
        try:
            filepath_with_ext = f"{filepath}.pkl" if not filepath.endswith('.pkl') else filepath
            
            if not os.path.exists(filepath_with_ext):
                raise FileNotFoundError(f"Database file not found: {filepath_with_ext}")
            
            with open(filepath_with_ext, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different data formats for backward compatibility
            if isinstance(data, dict):
                self.embeddings = data.get('embeddings', [])
                self.metadata = data.get('metadata', [])
                self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
            else:
                # Old format - assume it's just embeddings and metadata
                self.embeddings, self.metadata = data
            
            print(f"Database loaded from {filepath_with_ext} with {len(self.metadata)} segments")
            
        except Exception as e:
            print(f"Error loading database: {str(e)}")
            raise
    
    def __len__(self):
        """Return number of segments in database"""
        return len(self.metadata)
    
    def __str__(self):
        """String representation of database"""
        stats = self.get_statistics()
        return f"VideoSearchDatabase(segments={stats['total_segments']}, videos={stats['total_videos']}, dim={self.embedding_dim})"
