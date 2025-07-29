# embeddings.py
import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import os

class EmbeddingGenerator:
    def __init__(self, device: str = None):
        """Initialize embedding models for general video understanding"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP for visual embeddings
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load sentence transformer for text embeddings
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image"""
        try:
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                return np.zeros(512, dtype=np.float32)
            
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten().astype(np.float32)
            
        except Exception as e:
            print(f"Error generating image embedding for {image_path}: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            if not text or not text.strip():
                return np.zeros(384, dtype=np.float32)
            
            # Generate embedding for any text content
            embedding = self.text_model.encode(text)
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def generate_multimodal_embedding(self, image_path: str, text: str = "") -> np.ndarray:
        """Combine visual and text embeddings for any video content"""
        try:
            # Generate image embedding (512 dimensions)
            image_emb = self.generate_image_embedding(image_path)
            
            # Ensure image embedding is exactly 512 dimensions
            if len(image_emb) != 512:
                if len(image_emb) < 512:
                    padding = np.zeros(512 - len(image_emb), dtype=np.float32)
                    image_emb = np.concatenate([image_emb, padding])
                else:
                    image_emb = image_emb[:512]
            
            # Generate text embedding (384 dimensions)
            text_emb = self.generate_text_embedding(text)
            
            # Ensure text embedding is exactly 384 dimensions
            if len(text_emb) != 384:
                if len(text_emb) < 384:
                    padding = np.zeros(384 - len(text_emb), dtype=np.float32)
                    text_emb = np.concatenate([text_emb, padding])
                else:
                    text_emb = text_emb[:384]
            
            # Combine embeddings (512 + 384 = 896 dimensions)
            combined = np.concatenate([image_emb, text_emb])
            
            # Normalize the combined embedding
            normalized = combined / (np.linalg.norm(combined) + 1e-8)
            
            return normalized.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating multimodal embedding: {e}")
            return np.zeros(896, dtype=np.float32)
    
    def generate_query_embedding_for_search(self, query: str) -> np.ndarray:
        """Generate optimized embedding for any search query"""
        try:
            # Generate text embedding for the query
            text_embedding = self.text_model.encode(query).astype(np.float32)
            
            # Ensure correct text component size
            if len(text_embedding) != 384:
                if len(text_embedding) < 384:
                    padding = np.zeros(384 - len(text_embedding), dtype=np.float32)
                    text_embedding = np.concatenate([text_embedding, padding])
                else:
                    text_embedding = text_embedding[:384]
            
            # Create zero visual component for text-only queries
            visual_component = np.zeros(512, dtype=np.float32)
            
            # Combine to match stored multimodal embeddings
            query_embedding = np.concatenate([visual_component, text_embedding])
            
            # Normalize
            normalized = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            return normalized.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return np.zeros(896, dtype=np.float32)
