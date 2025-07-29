# audio_transcriber.py (Fixed version)
import whisper
from typing import List, Dict, Any, Union, cast
import os
import torch

class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model with proper device handling
        """
        try:
            # Clear any existing CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Whisper model '{model_size}' on {self.device}")
            
            self.model = whisper.load_model(model_size, device=self.device)
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            # Fallback to CPU
            self.device = "cpu"
            self.model = whisper.load_model(model_size, device="cpu")
    
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, Union[float, str]]]:
        """
        Transcribe audio file with enhanced error handling
        """
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return []
        
        try:
            print(f"Transcribing audio: {audio_path}")
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                print("Audio file is empty")
                return []
            
            print(f"Audio file size: {file_size / (1024*1024):.1f} MB")

            # Key fix in audio_transcriber.py transcribe_audio method:
            result = self.model.transcribe(
            audio_path,
            fp16=False,  # Disable FP16 to avoid tensor mismatches
            language="en",  # Specify language
            task="transcribe",
            verbose=False
            )
            
            segments = []
            
            # Cast to the expected type
            whisper_result = cast(Dict[str, Any], result)
            
            if 'segments' not in whisper_result:
                print("No segments found in transcription result")
                return []
            
            for segment in whisper_result['segments']:
                segment_dict = cast(Dict[str, Any], segment)
                
                # Validate segment data
                if all(key in segment_dict for key in ['start', 'end', 'text']):
                    segments.append({
                        'start': float(segment_dict['start']),
                        'end': float(segment_dict['end']),
                        'text': str(segment_dict['text']).strip(),
                        'confidence': 1.0 - float(segment_dict.get('no_speech_prob', 0.0))
                    })
            
            print(f"Successfully transcribed {len(segments)} segments")
            return segments
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            
            # Try with different options as fallback
            try:
                print("Trying fallback transcription method...")
                result = self.model.transcribe(
                    audio_path,
                    fp16=False,
                    task="transcribe",
                    verbose=False,
                    word_timestamps=False  # Disable word-level timestamps
                )
                
                # Simple extraction without detailed segments
                if 'text' in result and result['text'].strip():
                    return [{
                        'start': 0.0,
                        'end': 60.0,  # Assume 60-second chunks
                        'text': result['text'].strip(),
                        'confidence': 0.8
                    }]
                
            except Exception as fallback_error:
                print(f"Fallback transcription also failed: {fallback_error}")
            
            return []
