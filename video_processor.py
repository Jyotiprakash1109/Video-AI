# video_processor.py (Fixed version)
import cv2
import numpy as np
from typing import List, Tuple
import os
import re
import subprocess
from moviepy.editor import VideoFileClip

class VideoProcessor:
    def __init__(self, sampling_rate: int = 30):
        self.sampling_rate = sampling_rate
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to remove problematic characters"""
        # Remove or replace problematic characters
        filename = re.sub(r'[™®©]', '', filename)  # Remove trademark symbols
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)  # Replace invalid chars
        filename = re.sub(r'[^\w\-_. ]', '_', filename)  # Keep only safe characters
        filename = filename.strip()
        
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        
        return filename
    
    def extract_frames(self, video_path: str, output_dir: str) -> List[Tuple[str, float]]:
        """Extract frames from video at specified intervals with improved error handling"""
        try:
            # Sanitize the output directory name
            base_dir = os.path.dirname(output_dir)
            dir_name = os.path.basename(output_dir)
            sanitized_dir_name = self.sanitize_filename(dir_name)
            sanitized_output_dir = os.path.join(base_dir, sanitized_dir_name)
            
            print(f"Extracting frames to: {sanitized_output_dir}")
            
            # Create output directory
            if not os.path.exists(sanitized_output_dir):
                os.makedirs(sanitized_output_dir, exist_ok=True)
            
            # Verify video file exists
            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {video_path}")
                return []
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0:
                print("Error: Invalid FPS detected")
                cap.release()
                return []
            
            print(f"Video properties: FPS={fps:.2f}, Total frames={total_frames}")
            
            frame_count = 0
            extracted_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified intervals
                if frame_count % self.sampling_rate == 0:
                    timestamp = frame_count / fps
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(sanitized_output_dir, frame_filename)
                    
                    # Save frame with error checking
                    success = cv2.imwrite(frame_path, frame)
                    if success and os.path.exists(frame_path):
                        extracted_frames.append((frame_path, timestamp))
                    else:
                        print(f"Warning: Failed to save frame at {timestamp:.1f}s")
                
                frame_count += 1
            
            cap.release()
            
            print(f"Successfully extracted {len(extracted_frames)} frames")
            return extracted_frames
            
        except Exception as e:
            print(f"Error in frame extraction: {e}")
            if 'cap' in locals():
                cap.release()
            return []
    
    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio with multiple fallback methods"""
        try:
            print(f"Attempting audio extraction from: {video_path}")
            
            # Sanitize output path
            output_dir = os.path.dirname(output_path)
            output_filename = os.path.basename(output_path)
            sanitized_filename = self.sanitize_filename(output_filename)
            sanitized_output_path = os.path.join(output_dir, sanitized_filename)
            
            # Method 1: Try MoviePy first
            try:
                print("Trying MoviePy extraction...")
                video = VideoFileClip(video_path)
                
                if video.audio is None:
                    print("No audio track found in video")
                    video.close()
                    return None
                
                # Fix the MoviePy method - remove temp_audiofile parameter
                video.audio.write_audiofile(
                    sanitized_output_path,
                    codec='pcm_s16le',
                    ffmpeg_params=['-ar', '16000', '-ac', '1'],
                    verbose=False,
                    logger=None
                )
                
                video.close()
                print(f"✅ MoviePy extraction successful: {sanitized_output_path}")
                return sanitized_output_path
                
            except Exception as moviepy_error:
                print(f"MoviePy failed: {moviepy_error}")
                if 'video' in locals():
                    video.close()
            
            # Method 2: Try direct FFmpeg
            print("Trying direct FFmpeg extraction...")
            return self._extract_audio_ffmpeg(video_path, sanitized_output_path)
            
        except Exception as e:
            print(f"All audio extraction methods failed: {e}")
            return None
    
    def _extract_audio_ffmpeg(self, video_path: str, output_path: str) -> str:
        """Extract audio using FFmpeg directly"""
        try:
            # Test different FFmpeg approaches
            ffmpeg_commands = [
                # Standard approach
                [
                    'ffmpeg', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1',
                    '-y', output_path
                ],
                # Alternative approach
                [
                    'ffmpeg', '-i', video_path,
                    '-vn', '-acodec', 'libmp3lame',
                    '-ar', '16000', '-ac', '1',
                    '-y', output_path.replace('.wav', '.mp3')
                ]
            ]
            
            for i, cmd in enumerate(ffmpeg_commands):
                try:
                    print(f"Trying FFmpeg method {i+1}...")
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        final_output = cmd[-1]
                        if os.path.exists(final_output):
                            print(f"✅ FFmpeg extraction successful: {final_output}")
                            return final_output
                    else:
                        print(f"FFmpeg method {i+1} failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"FFmpeg method {i+1} timed out")
                except Exception as cmd_error:
                    print(f"FFmpeg method {i+1} error: {cmd_error}")
            
            return None
            
        except Exception as e:
            print(f"FFmpeg extraction failed: {e}")
            return None
