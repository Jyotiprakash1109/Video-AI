# check.py (Final version with delete functionality)
import streamlit as st
import os
from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber
from embeddings import EmbeddingGenerator
from vector_database_fixed import VideoSearchDatabase
from query_processor import QueryProcessor
from video_indexer import VideoIndexer

class VideoSearchApp:
    def __init__(self):
        pass
    
    @st.cache_resource
    def setup_components(_self):
        """Initialize all components for general video processing"""
        try:
            video_processor = VideoProcessor(sampling_rate=30)
            audio_transcriber = AudioTranscriber(model_size="base")
            embedding_generator = EmbeddingGenerator()
            search_database = VideoSearchDatabase(embedding_dim=896)
            query_processor = QueryProcessor(embedding_generator, search_database)
            video_indexer = VideoIndexer(
                video_processor, audio_transcriber, 
                embedding_generator, search_database
            )
            
            return {
                'video_processor': video_processor,
                'audio_transcriber': audio_transcriber,
                'embedding_generator': embedding_generator,
                'search_database': search_database,
                'query_processor': query_processor,
                'video_indexer': video_indexer
            }
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return None
    
    def run(self):
        st.set_page_config(
            page_title="Video NLP Search Engine",
            page_icon="🎥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎥 Universal Video NLP Search Engine")
        st.markdown("**Search any video content using natural language queries with AI-powered semantic understanding**")
        
        components = self.setup_components()
        if not components:
            st.error("Failed to initialize components. Please check your dependencies.")
            return
        
        # Sidebar navigation
        st.sidebar.title("🧭 Navigation")
        st.sidebar.markdown("---")
        
        page = st.sidebar.selectbox(
            "Choose a page", 
            ["🔍 Search Videos", "📹 Index New Video", "📊 Database Stats", "🗑️ Delete Videos", "💡 Help & Examples"],
            index=0
        )
        
        # Show current database status in sidebar
        if os.path.exists("video_database.pkl"):
            try:
                temp_db = VideoSearchDatabase()
                temp_db.load("video_database")
                st.sidebar.success(f"✅ Database Active: {len(temp_db.metadata)} segments")
            except:
                st.sidebar.warning("⚠️ Database file exists but may be corrupted")
        else:
            st.sidebar.warning("❌ No database found")
        
        # Route to appropriate page
        if page == "🔍 Search Videos":
            self.search_page(components)
        elif page == "📹 Index New Video":
            self.index_page(components)
        elif page == "📊 Database Stats":
            self.stats_page(components)
        elif page == "🗑️ Delete Videos":
            self.delete_page(components)
        elif page == "💡 Help & Examples":
            self.help_page()
    
    def search_page(self, components):
        st.header("🔍 Search Videos with Natural Language")
        
        # Load existing database if available
        if os.path.exists("video_database.pkl"):
            try:
                components['search_database'].load("video_database")
                database_size = len(components['search_database'].metadata)
                
                # Show database info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Segments", database_size)
                with col2:
                    stats = components['search_database'].get_statistics()
                    st.metric("Total Videos", stats['total_videos'])
                with col3:
                    avg_segments = database_size / max(stats['total_videos'], 1)
                    st.metric("Avg Segments/Video", f"{avg_segments:.0f}")
                
            except Exception as e:
                st.error(f"Error loading database: {str(e)}")
                st.stop()
        else:
            st.warning("No indexed videos found. Please index some videos first using the 'Index New Video' page.")
            st.stop()
        
        # Search interface
        st.markdown("### 🎯 Enter Your Search Query")
        
        # Example queries section
        with st.expander("💡 See Example Queries", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🏏 Sports Content:**
                - "player hitting six"
                - "celebration after goal"
                - "coach giving instructions"
                - "crowd cheering"
                """)
            with col2:
                st.markdown("""
                **🎬 General Content:**
                - "person speaking at podium"
                - "car driving fast"
                - "people laughing together"
                - "music playing"
                """)
        
        # Main search input
        query = st.text_input(
            "Search Query",
            placeholder="e.g., virat batting, person jumping, music playing, people talking...",
            help="Describe what you want to see using natural language",
            key="main_search",
            label_visibility="collapsed"
        )
        
        # Search configuration
        st.markdown("### ⚙️ Search Settings")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_results = st.slider("Max Results", 1, 20, 10, help="Maximum number of clips to show")
        with col2:
            min_score = st.slider("Min Relevance", 0.0, 1.0, 0.2, 0.05, help="Minimum similarity score (lower = more results)")
        with col3:
            use_context = st.checkbox("Context Enhancement", value=True, help="Consider nearby segments for better results")
        with col4:
            show_frames = st.checkbox("Show Frame Previews", value=True, help="Display video frame thumbnails")
        
        if query:
            with st.spinner("🔍 Searching for relevant clips..."):
                try:
                    # Search based on settings
                    if use_context:
                        raw_results = components['query_processor'].enhance_search_with_context(
                            query, k=max_results*2
                        )
                    else:
                        raw_results = components['query_processor'].search_segments(query, k=max_results*2)
                    
                    # Filter results
                    filtered_results = components['query_processor'].filter_by_logic(
                        query, raw_results, min_score=min_score
                    )
                    
                    # Display search results
                    if filtered_results:
                        st.success(f"🎯 Found {len(filtered_results)} relevant clips for: **'{query}'**")
                        
                        # Sort by relevance score
                        filtered_results.sort(key=lambda x: x[1], reverse=True)
                        
                        # Display results
                        for i, (segment, score) in enumerate(filtered_results[:max_results]):
                            # Create an attractive result card
                            with st.container():
                                st.markdown(f"### 🎬 Clip {i+1} - Relevance: {score:.1%}")
                                
                                col1, col2 = st.columns([1, 2] if show_frames else [0.1, 2])
                                
                                if show_frames:
                                    with col1:
                                        # Video information
                                        st.markdown(f"**📹 Video:** {segment['video_id']}")
                                        st.markdown(f"**⏰ Time:** {segment['timestamp']:.1f}s - {segment['timestamp'] + segment.get('duration', 5.0):.1f}s")
                                        st.markdown(f"**⏱️ Duration:** {segment.get('duration', 5.0):.1f}s")
                                        
                                        # Show frame thumbnail
                                        if 'frame_path' in segment and os.path.exists(segment['frame_path']):
                                            st.image(
                                                segment['frame_path'], 
                                                caption=f"Frame at {segment['timestamp']:.1f}s",
                                                width=200
                                            )
                                        else:
                                            st.info("🖼️ Frame preview not available")
                                
                                with col2:
                                    # Transcript/Audio content
                                    st.markdown("**🎤 Audio Transcript:**")
                                    transcript = segment.get('transcript', '').strip()
                                    
                                    if transcript:
                                        # Highlight query terms in transcript
                                        highlighted_transcript = self.highlight_query_terms(transcript, query)
                                        st.markdown(f"*\"{highlighted_transcript}\"*")
                                        
                                        # Show confidence if available
                                        confidence = segment.get('confidence', 0)
                                        if confidence > 0:
                                            st.caption(f"Transcription confidence: {confidence:.1%}")
                                    else:
                                        st.info("*Visual content only - no audio transcript available*")
                                    
                                    # Video player
                                    if 'video_path' in segment and os.path.exists(segment['video_path']):
                                        try:
                                            st.video(segment['video_path'], start_time=int(segment['timestamp']))
                                        except Exception as video_error:
                                            st.error(f"❌ Could not load video player: {str(video_error)}")
                                            st.info("💡 Try downloading the video file to play locally")
                                    else:
                                        st.warning("🎥 Video file not found")
                                
                                st.markdown("---")  # Separator between results
                    else:
                        # No results found
                        st.warning("🔍 No results found for your query.")
                        
                        # Suggestions for better results
                        st.markdown("### 💡 Try These Tips:")
                        suggestions_col1, suggestions_col2 = st.columns(2)
                        
                        with suggestions_col1:
                            st.markdown("""
                            **🎯 Improve Your Search:**
                            - Lower the minimum relevance score
                            - Use more general terms (e.g., 'person' instead of specific names)
                            - Describe visual actions or scenes
                            - Try single keywords first
                            """)
                        
                        with suggestions_col2:
                            st.markdown("""
                            **🔧 Technical Tips:**
                            - Check if your video contains the searched content
                            - Enable context enhancement for better matching
                            - Try variations of your query
                            - Use action words (running, talking, playing)
                            """)
                
                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")
                    
                    with st.expander("🔧 Debug Information"):
                        st.code(f"""
                        Query: {query}
                        Database segments: {len(components['search_database'].metadata)}
                        Embedding dimension: {components['search_database'].embedding_dim}
                        Error: {str(e)}
                        """)
    
    def index_page(self, components):
        st.header("📹 Index New Video")
        
        st.markdown("""
        Upload any type of video to make it searchable with natural language queries. 
        The system will extract both visual frames and audio transcripts for comprehensive search capabilities.
        """)
        
        # Supported formats info
        with st.expander("📋 Supported Video Formats & Requirements"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **✅ Supported Formats:**
                - MP4 (recommended)
                - AVI, MOV, MKV
                - WebM, FLV
                - Most common video codecs
                """)
            with col2:
                st.markdown("""
                **⚡ Processing Requirements:**
                - Video with clear audio for best results
                - Recommended: Under 1GB file size
                - Processing time: ~1-2 minutes per minute of video
                """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'],
            help="Select a video file from your computer"
        )
        
        if uploaded_file is not None:
            # Display file information
            file_size_mb = uploaded_file.size / (1024*1024)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 File Name", uploaded_file.name)
            with col2:
                st.metric("📊 File Size", f"{file_size_mb:.1f} MB")
            with col3:
                estimated_time = max(1, int(file_size_mb / 10))  # Rough estimate
                st.metric("⏱️ Est. Processing Time", f"~{estimated_time} min")
            
            # Save uploaded file
            if not os.path.exists("temp"):
                os.makedirs("temp")
            
            video_path = os.path.join("temp", uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Video ID input
            video_id = st.text_input(
                "Enter a unique video identifier:", 
                value=uploaded_file.name.split('.')[0],
                help="This ID will be used to identify your video in search results",
                max_chars=50
            )
            
            # Processing options
            st.markdown("### 🛠️ Processing Options")
            col1, col2 = st.columns(2)
            
            with col1:
                frame_rate = st.selectbox(
                    "Frame Sampling Rate", 
                    [15, 30, 45, 60], 
                    index=1,
                    help="Lower values = faster processing, higher values = more detailed indexing"
                )
            
            with col2:
                audio_model = st.selectbox(
                    "Audio Model Size",
                    ["tiny", "base", "small", "medium"],
                    index=1,
                    help="Larger models are more accurate but slower"
                )
            
            # Index button
            if st.button("🚀 Start Video Indexing", type="primary"):
                if not video_id.strip():
                    st.error("Please provide a video ID")
                    return
                
                # Check if video ID already exists
                if os.path.exists("video_database.pkl"):
                    try:
                        temp_db = VideoSearchDatabase()
                        temp_db.load("video_database")
                        existing_videos = [seg['video_id'] for seg in temp_db.metadata]
                        if video_id in existing_videos:
                            if not st.checkbox(f"Video ID '{video_id}' already exists. Overwrite?"):
                                st.warning("Please choose a different video ID or check the overwrite option.")
                                return
                    except:
                        pass
                
                # Start processing
                with st.spinner("🎬 Processing video... This may take several minutes depending on video length"):
                    try:
                        # Create progress tracking
                        progress_container = st.container()
                        
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Update components with user settings
                            components['video_processor'].sampling_rate = frame_rate
                            components['audio_transcriber'] = AudioTranscriber(model_size=audio_model)
                            
                            status_text.text("🎬 Starting video processing...")
                            progress_bar.progress(5)
                            
                            # Index the video
                            success = components['video_indexer'].index_video(
                                video_path, video_id, temp_dir="temp"
                            )
                            
                            progress_bar.progress(90)
                            
                            if success:
                                status_text.text("💾 Saving to searchable database...")
                                components['search_database'].save("video_database")
                                progress_bar.progress(100)
                                
                                # Success message
                                st.success("✅ Video indexed successfully!")
                                
                                # Show results
                                database_size = len(components['search_database'].metadata)
                                stats = components['search_database'].get_statistics()
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("📊 Total Segments", database_size)
                                with col2:
                                    st.metric("🎥 Total Videos", stats['total_videos'])
                                with col3:
                                    segments_this_video = sum(1 for seg in components['search_database'].metadata if seg['video_id'] == video_id)
                                    st.metric("📹 This Video Segments", segments_this_video)
                                
                                # Celebration and next steps
                                st.balloons()
                                
                                st.markdown("### 🎉 What's Next?")
                                st.info("Your video is now fully searchable! Go to the 'Search Videos' page to try natural language queries.")
                                
                                # Sample search suggestions
                                with st.expander("💡 Suggested Search Queries for Your Video"):
                                    st.markdown("""
                                    Try searching for:
                                    - Actions you saw in the video
                                    - People or objects that appeared
                                    - Sounds or speech you heard
                                    - Emotions or scenes depicted
                                    """)
                                
                            else:
                                st.error("❌ Failed to index video.")
                                st.markdown("### 🔧 Troubleshooting:")
                                st.markdown("""
                                - Check if the video file is not corrupted
                                - Ensure the video has readable audio/video streams
                                - Try with a different video format (MP4 recommended)
                                - Check if you have sufficient disk space
                                """)
                    
                    except Exception as e:
                        st.error(f"❌ Error during indexing: {str(e)}")
                        
                        with st.expander("🔧 Error Details"):
                            st.code(f"""
                            Video Path: {video_path}
                            Video ID: {video_id}
                            Error: {str(e)}
                            
                            Common solutions:
                            1. Check FFmpeg installation
                            2. Verify video file integrity
                            3. Ensure sufficient disk space
                            4. Try a simpler video format
                            """)
                
                # Cleanup uploaded file
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                except:
                    pass  # Don't fail if cleanup doesn't work
    
    def stats_page(self, components):
        st.header("📊 Database Statistics & Analytics")
        
        if os.path.exists("video_database.pkl"):
            try:
                components['search_database'].load("video_database")
                stats = components['search_database'].get_statistics()
                
                # Overview metrics
                st.markdown("### 📈 Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎥 Total Videos", stats['total_videos'])
                with col2:
                    st.metric("📹 Total Segments", stats['total_segments'])
                with col3:
                    avg_segments = stats['total_segments'] / max(stats['total_videos'], 1)
                    st.metric("📊 Avg Segments/Video", f"{avg_segments:.1f}")
                with col4:
                    st.metric("🔢 Embedding Dimension", stats['embedding_dimension'])
                
                # Video breakdown
                if stats['videos']:
                    st.markdown("### 🎬 Video Collection")
                    
                    # Create a nice table view
                    video_data = []
                    for video_id, segment_count in stats['videos'].items():
                        # Calculate approximate duration (segments * 5 seconds / 60)
                        approx_duration = (segment_count * 5) / 60
                        video_data.append({
                            "Video ID": video_id,
                            "Segments": segment_count,
                            "Approx Duration": f"{approx_duration:.1f} min"
                        })
                    
                    st.dataframe(video_data, use_container_width=True)
                
                # Sample content preview
                if components['search_database'].metadata:
                    st.markdown("### 🔍 Sample Content Preview")
                    
                    # Show sample segments from different videos
                    sample_segments = components['search_database'].metadata[:5]
                    
                    for i, segment in enumerate(sample_segments):
                        with st.expander(f"📄 Sample {i+1}: {segment['video_id']} @ {segment['timestamp']:.1f}s"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if 'frame_path' in segment and os.path.exists(segment['frame_path']):
                                    st.image(segment['frame_path'], width=200)
                                else:
                                    st.info("🖼️ Frame not available")
                            
                            with col2:
                                transcript = segment.get('transcript', 'Visual content only')
                                if transcript.strip():
                                    st.markdown(f"**Transcript:** *{transcript}*")
                                else:
                                    st.markdown("**Content:** Visual-only segment")
                                
                                st.markdown(f"**Timestamp:** {segment['timestamp']:.1f}s")
                                st.markdown(f"**Duration:** {segment.get('duration', 5.0):.1f}s")
                
                # Database health check
                st.markdown("### 🏥 Database Health")
                
                # Check for issues
                issues = []
                segments_with_audio = sum(1 for seg in components['search_database'].metadata if seg.get('transcript', '').strip())
                segments_with_frames = sum(1 for seg in components['search_database'].metadata if 'frame_path' in seg)
                
                if segments_with_audio == 0:
                    issues.append("⚠️ No audio transcripts found - search will be visual-only")
                if segments_with_frames < stats['total_segments']:
                    issues.append("⚠️ Some segments missing frame references")
                
                if issues:
                    for issue in issues:
                        st.warning(issue)
                else:
                    st.success("✅ Database appears healthy!")
                
                # Additional stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎤 Segments with Audio", segments_with_audio)
                with col2:
                    st.metric("🖼️ Segments with Frames", segments_with_frames)
                
            except Exception as e:
                st.error(f"❌ Error loading database statistics: {str(e)}")
                
                with st.expander("🔧 Debug Information"):
                    st.code(f"""
                    Error details: {str(e)}
                    Database file exists: {os.path.exists('video_database.pkl')}
                    
                    Try:
                    1. Re-index your videos if database is corrupted
                    2. Check file permissions
                    3. Ensure sufficient disk space
                    """)
        else:
            st.info("📭 No database found. Index some videos first to see statistics!")
            
            st.markdown("### 🚀 Get Started")
            st.markdown("""
            1. Go to **Index New Video** page
            2. Upload a video file
            3. Wait for processing to complete
            4. Return here to see statistics
            5. Use **Search Videos** to find content
            """)
    
    def delete_page(self, components):
        st.header("🗑️ Delete Indexed Videos")
        
        if not os.path.exists("video_database.pkl"):
            st.warning("📭 No database found. Nothing to delete!")
            st.markdown("### 🚀 Get Started")
            st.info("Go to the 'Index New Video' page to upload and index videos first.")
            return
        
        try:
            # Load database
            components['search_database'].load("video_database")
            stats = components['search_database'].get_statistics()
            
            if stats['total_videos'] == 0:
                st.info("📭 Database is empty!")
                return
            
            st.markdown("### 🎬 Current Videos in Database")
            
            # Show current videos with details
            video_data = []
            for video_id, segment_count in stats['videos'].items():
                approx_duration = (segment_count * 5) / 60
                video_data.append({
                    "Video ID": video_id,
                    "Segments": segment_count,
                    "Approx Duration": f"{approx_duration:.1f} min"
                })
            
            st.dataframe(video_data, use_container_width=True)
            
            # Deletion options
            st.markdown("### 🗑️ Deletion Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Delete Specific Video")
                selected_video = st.selectbox(
                    "Choose video to delete:",
                    options=list(stats['videos'].keys()),
                    help="Select a video to remove from the database"
                )
                
                if st.button("🗑️ Delete Selected Video", type="secondary"):
                    if st.session_state.get('confirm_delete_specific', False):
                        # Perform deletion
                        segments_removed = components['search_database'].remove_video(selected_video)
                        components['search_database'].save("video_database")
                        
                        st.success(f"✅ Deleted '{selected_video}' ({segments_removed} segments)")
                        st.session_state['confirm_delete_specific'] = False
                        st.rerun()
                    else:
                        st.warning(f"⚠️ Click again to confirm deletion of '{selected_video}'")
                        st.session_state['confirm_delete_specific'] = True
            
            with col2:
                st.markdown("#### Delete Entire Database")
                st.warning("⚠️ This will remove ALL indexed videos!")
                
                if st.button("🗑️ Delete All Videos", type="secondary"):
                    if st.session_state.get('confirm_delete_all', False):
                        # Delete entire database
                        os.remove("video_database.pkl")
                        st.success("✅ Entire database deleted!")
                        st.session_state['confirm_delete_all'] = False
                        st.rerun()
                    else:
                        st.error("⚠️ Click again to confirm deletion of ALL videos")
                        st.session_state['confirm_delete_all'] = True
            
            # Additional options
            st.markdown("### 🧹 Additional Cleanup Options")
            
            if st.button("🧽 Clean Temporary Files", type="secondary"):
                temp_cleaned = 0
                temp_locations = ["temp", "Extracted Frames"]
                
                for location in temp_locations:
                    try:
                        if os.path.exists(location):
                            if os.path.isdir(location):
                                import shutil
                                shutil.rmtree(location)
                                temp_cleaned += 1
                            elif os.path.isfile(location):
                                os.remove(location)
                                temp_cleaned += 1
                    except Exception as e:
                        st.warning(f"Could not clean {location}: {e}")
                
                if temp_cleaned > 0:
                    st.success(f"🧹 Cleaned {temp_cleaned} temporary locations")
                else:
                    st.info("✨ No temporary files found to clean")
        
        except Exception as e:
            st.error(f"❌ Error loading database: {e}")
            st.markdown("### 🔧 Manual Cleanup")
            st.code("""
            If the database is corrupted, you can manually delete:
            1. video_database.pkl (main database file)
            2. temp/ directory (temporary files)
            3. Extracted Frames/ directory (frame cache)
            """)
    
    def help_page(self):
        st.header("💡 Help & Examples")
        
        # How it works section
        st.markdown("### 🧠 How Video NLP Search Works")
        
        with st.expander("🔍 Understanding the Search Process", expanded=True):
            st.markdown("""
            **Your Video NLP Search Model combines multiple AI technologies:**
            
            1. **🎬 Visual Understanding** - Analyzes video frames to understand scenes, objects, and actions
            2. **🎤 Audio Processing** - Transcribes speech and understands audio content
            3. **🧠 Semantic Matching** - Matches your natural language queries with video content meaning
            4. **⚡ Smart Ranking** - Ranks results by relevance and context
            """)
        
        # Search tips
        st.markdown("### 🎯 Search Tips & Best Practices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Good Search Queries:**
            - "person running in park"
            - "car driving at night"
            - "people laughing together"
            - "music performance on stage"
            - "cooking in kitchen"
            - "sports celebration"
            """)
        
        with col2:
            st.markdown("""
            **❌ Less Effective Queries:**
            - Very specific names (unless in audio)
            - Complex technical terms
            - Abstract concepts
            - Overly long sentences
            - Misspelled words
            """)
        
        # Query examples by category
        st.markdown("### 📚 Query Examples by Category")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🏃 Actions", "👥 People", "🎵 Audio", "🌟 Emotions"])
        
        with tab1:
            st.markdown("""
            **Action-Based Searches:**
            - "person jumping"
            - "car turning corner"
            - "dog running"
            - "cooking food"
            - "writing on board"
            - "playing instrument"
            """)
        
        with tab2:
            st.markdown("""
            **People-Focused Searches:**
            - "person speaking"
            - "child playing"
            - "group discussion"
            - "teacher explaining"
            - "audience clapping"
            - "person waving"
            """)
        
        with tab3:
            st.markdown("""
            **Audio-Based Searches:**
            - "music playing"
            - "person talking"
            - "crowd cheering"
            - "phone ringing"
            - "door closing"
            - "applause"
            """)
        
        with tab4:
            st.markdown("""
            **Emotion & Scene Searches:**
            - "celebration moment"
            - "sad expression"
            - "excitement"
            - "peaceful scene"
            - "tense moment"
            - "happy gathering"
            """)
        
        # Technical details
        st.markdown("### 🔧 Technical Details")
        
        with st.expander("⚙️ System Architecture"):
            st.markdown("""
            **Core Components:**
            - **CLIP Model**: Understands visual content and connects it with language
            - **Whisper**: Transcribes audio with high accuracy
            - **Sentence Transformers**: Creates semantic embeddings for text
            - **Vector Database**: Enables fast similarity search across video segments
            - **Multimodal Fusion**: Combines visual and audio understanding
            """)
        
        with st.expander("📊 Performance Metrics"):
            st.markdown("""
            **What the scores mean:**
            - **Relevance Score**: 0.0 to 1.0 (higher = more relevant)
            - **0.8-1.0**: Excellent match
            - **0.6-0.8**: Good match
            - **0.4-0.6**: Moderate match
            - **0.2-0.4**: Weak match
            - **Below 0.2**: Poor match (filtered out by default)
            """)
        
        # Troubleshooting
        st.markdown("### 🔧 Troubleshooting")
        
        with st.expander("❓ Common Issues & Solutions"):
            st.markdown("""
            **Problem: No search results found**
            - Lower the minimum relevance score
            - Try more general terms
            - Check if your video actually contains the searched content
            - Enable context enhancement
            
            **Problem: Poor search quality**
            - Ensure your video has clear audio
            - Use descriptive action words
            - Try multiple related queries
            - Check video processing completed successfully
            
            **Problem: Slow performance**
            - Reduce maximum results
            - Use more specific queries
            - Consider re-indexing with lower frame sampling rate
            """)
        
        # Contact info
        st.markdown("### 📧 Support & Feedback")
        st.info("""
        This AI-powered video search system demonstrates cutting-edge multimodal AI capabilities.
        Perfect for applications in media, education, sports analysis, and content management.
        """)
    
    def highlight_query_terms(self, text: str, query: str) -> str:
        """Highlight query terms in the transcript"""
        try:
            import re
            query_words = query.lower().split()
            highlighted_text = text
            
            for word in query_words:
                if len(word) > 2:  # Only highlight words longer than 2 characters
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    highlighted_text = pattern.sub(f"**{word}**", highlighted_text)
            
            return highlighted_text
        except:
            return text

if __name__ == "__main__":
    app = VideoSearchApp()
    app.run()
