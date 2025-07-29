# video_indexer.py (Enhanced with duplicate prevention)
import os
import time
import threading
from typing import List, Dict
import json
import traceback

class VideoIndexer:
    def __init__(self, video_processor, audio_transcriber, embedding_generator, search_database):
        self.video_processor = video_processor
        self.audio_transcriber = audio_transcriber
        self.embedding_generator = embedding_generator
        self.search_database = search_database
        self._processing_lock = threading.Lock()  # Prevent concurrent processing
        self._currently_processing = set()  # Track videos being processed
    
    def index_video(self, video_path: str, video_id: str, temp_dir: str = "temp") -> bool:
        """Index video with duplicate processing prevention"""
        
        # Prevent duplicate processing
        with self._processing_lock:
            if video_id in self._currently_processing:
                print(f"Video {video_id} is already being processed. Skipping duplicate.")
                return False
            self._currently_processing.add(video_id)
        
        try:
            return self._index_video_internal(video_path, video_id, temp_dir)
        finally:
            # Always remove from processing set
            with self._processing_lock:
                self._currently_processing.discard(video_id)
    
    def _index_video_internal(self, video_path: str, video_id: str, temp_dir: str = "temp") -> bool:
        """Internal indexing method"""
        try:
            print(f"Indexing video: {video_path}")
            
            # Sanitize video_id for safe file operations
            safe_video_id = self.video_processor.sanitize_filename(video_id)
            
            # Validate input file
            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {video_path}")
                return False
            
            # Create temporary directories with unique names to avoid conflicts
            timestamp = int(time.time())
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
            
            frames_dir = os.path.join(temp_dir, f"{safe_video_id}_frames_{timestamp}")
            audio_path = os.path.join(temp_dir, f"{safe_video_id}_audio_{timestamp}.wav")
            
            # Extract frames
            print("Extracting frames...")
            frames = self.video_processor.extract_frames(video_path, frames_dir)
            
            if not frames:
                print("Error: No frames extracted from video")
                return False
            
            print(f"Extracted {len(frames)} frames successfully")
            
            # Extract audio with error handling
            print("Extracting audio...")
            audio_extracted = self.video_processor.extract_audio(video_path, audio_path)
            
            transcription_segments = []
            if audio_extracted and os.path.exists(audio_extracted):
                print("Transcribing audio...")
                transcription_segments = self.audio_transcriber.transcribe_audio(audio_extracted)
                print(f"Transcribed {len(transcription_segments)} audio segments")
            else:
                print("Warning: No audio extracted, continuing with visual-only processing")
            
            # Create segment mappings
            print("Creating embeddings...")
            segments_data = []
            embeddings = []
            
            processed_count = 0
            for frame_path, timestamp_val in frames:
                try:
                    # Double-check frame file exists
                    if not os.path.exists(frame_path):
                        print(f"Warning: Frame file not found: {frame_path}")
                        continue
                    
                    # Find corresponding transcript
                    transcript_text = ""
                    if transcription_segments:
                        for trans_seg in transcription_segments:
                            if trans_seg['start'] <= timestamp_val <= trans_seg['end']:
                                transcript_text = trans_seg['text'].strip()
                                break
                    
                    # Generate multimodal embedding
                    embedding = self.embedding_generator.generate_multimodal_embedding(
                        frame_path, transcript_text
                    )
                    
                    # Validate embedding
                    if embedding is None or len(embedding) == 0:
                        print(f"Warning: Failed to generate embedding for frame at {timestamp_val:.1f}s")
                        continue
                    
                    # Store segment metadata
                    segment_data = {
                        'video_id': video_id,
                        'video_path': video_path,
                        'timestamp': timestamp_val,
                        'frame_path': frame_path,
                        'transcript': transcript_text,
                        'duration': 5.0
                    }
                    
                    segments_data.append(segment_data)
                    embeddings.append(embedding)
                    processed_count += 1
                    
                    # Progress indicator
                    if processed_count % 50 == 0:
                        print(f"Processed {processed_count}/{len(frames)} frames...")
                
                except Exception as frame_error:
                    print(f"Error processing frame at {timestamp_val}s: {frame_error}")
                    continue
            
            if not embeddings:
                print("Error: No embeddings generated")
                self._cleanup_temp_files(frames_dir, audio_extracted or audio_path)
                return False
            
            # Add to search database
            print(f"Adding {len(embeddings)} segments to search database...")
            self.search_database.add_segments(embeddings, segments_data)
            
            # Wait a moment before cleanup to ensure all file operations complete
            time.sleep(1)
            
            # Cleanup temporary files
            self._cleanup_temp_files(frames_dir, audio_extracted or audio_path)
            
            print(f"Successfully indexed video: {video_id}")
            print(f"Total segments processed: {len(segments_data)}")
            return True
            
        except Exception as e:
            print(f"Error indexing video {video_path}: {str(e)}")
            print("Full error traceback:")
            traceback.print_exc()
            
            # Cleanup on error
            try:
                timestamp = int(time.time())
                frames_dir = os.path.join(temp_dir, f"{safe_video_id}_frames_{timestamp}")
                audio_path = os.path.join(temp_dir, f"{safe_video_id}_audio_{timestamp}.wav")
                self._cleanup_temp_files(frames_dir, audio_path)
            except:
                pass
            
            return False
    
    def _cleanup_temp_files(self, frames_dir: str, audio_path: str):
        """Enhanced cleanup with retry mechanism"""
        # Wait a bit to ensure all file handles are closed
        time.sleep(2)
        
        # Cleanup frames directory
        try:
            if os.path.exists(frames_dir):
                import shutil
                
                # Try multiple times with increasing delays
                for attempt in range(3):
                    try:
                        shutil.rmtree(frames_dir)
                        print(f"Cleaned up frames directory: {frames_dir}")
                        break
                    except PermissionError:
                        if attempt < 2:
                            print(f"Cleanup attempt {attempt + 1} failed, retrying...")
                            time.sleep(5)  # Wait longer between attempts
                        else:
                            print(f"Warning: Could not remove frames directory {frames_dir}")
                            print("You may need to manually delete this directory later.")
        except Exception as e:
            print(f"Warning: Error during frames cleanup: {e}")
        
        # Cleanup audio file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"Cleaned up audio file: {audio_path}")
        except Exception as e:
            print(f"Warning: Could not remove audio file {audio_path}: {e}")
