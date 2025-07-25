import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
from transformers import pipeline
import time
from typing import List, Optional, Dict, Any

# ============ Utility Functions ============

def get_video_id(youtube_url: str) -> str:
    """Extract video ID from YouTube URL.
    
    Args:
        youtube_url: Full YouTube URL
        
    Returns:
        str: Video ID
        
    Raises:
        ValueError: If URL is invalid
    """
    if not youtube_url:
        raise ValueError("Please enter a YouTube URL.")
        
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',  # Standard URL
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed URL
        r'(?:youtu.be\/)([0-9A-Za-z_-]{11})'  # Short URL
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
            
    raise ValueError("Invalid YouTube URL. Please enter a valid YouTube video URL.")

def clean_transcript(text: str) -> str:
    """Clean and preprocess transcript text.
    
    Args:
        text: Raw transcript text
        
    Returns:
        str: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Remove bracketed actions/sounds like [Music], [Applause], etc.
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Remove timestamps if any
    text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '', text)
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    # Remove repeated lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    seen = set()
    cleaned_lines = []
    
    for line in lines:
        if line not in seen:
            seen.add(line)
            cleaned_lines.append(line)
            
    return ' '.join(cleaned_lines)

def chunk_text(text: str, max_len: int = 1800) -> List[str]:
    """Split text into chunks of maximum length.
    
    Args:
        text: Input text to chunk
        max_len: Maximum length of each chunk
        
    Returns:
        List[str]: List of text chunks
    """
    if not text:
        return []
        
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_len:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="MBZUAI/LaMini-Flan-T5-248M")

# =========== Streamlit UI ===========

# 🎬 YouTube Video Transcript Summarizer

# Set page config (must be the first Streamlit command)
st.set_page_config(
    page_title="YouTube Transcript Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            max-width: 900px;
            padding: 2rem;
        }
        .stTextArea [data-baseweb=base-input] {
            min-height: 200px;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with app info
with st.sidebar:
    st.title("About")
    st.markdown("""
    This app helps you to:
    - Extract transcripts from YouTube videos
    - Generate AI-powered summaries
    - Save time by getting key points from long videos
    
    Simply paste a YouTube URL and choose your options below.
    """)
    
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Paste a YouTube video URL
    2. Toggle summarization if desired
    3. View the transcript and summary
    """)

# Main content
st.title("🎬 YouTube Video Transcript Summarizer")
st.markdown("Paste a YouTube video link below to extract the transcript and generate an AI-powered summary.")

# Input section
with st.form("youtube_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    with col2:
        st.markdown("##")  # For vertical alignment
        summarize = st.checkbox("Generate AI Summary", value=True, help="Enable to generate an AI-powered summary")
    
    submitted = st.form_submit_button("Process Video")

if submitted and youtube_url:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Extract video ID
        status_text.text("🔍 Extracting video ID...")
        progress_bar.progress(10)
        video_id = get_video_id(youtube_url)
        
        # Step 2: Fetch transcript
        status_text.text("📥 Fetching transcript...")
        progress_bar.progress(30)
        try:
            # Get the transcript using the current youtube-transcript-api interface
            try:
                # Create a YouTubeTranscriptApi instance
                yt_transcript = YouTubeTranscriptApi()
                # Get the transcript data
                transcript_data = yt_transcript.fetch(video_id)
                # Convert to raw data and extract text
                transcript_text = "\n".join([item['text'] for item in transcript_data.to_raw_data()])
            except Exception as e:
                if "No transcript found" in str(e) or "Could not retrieve a transcript" in str(e):
                    raise NoTranscriptFound("No English transcript is available for this video or the video doesn't have captions.")
                elif "TranscriptsDisabled" in str(e):
                    raise TranscriptsDisabled("Transcripts are disabled for this video.")
                else:
                    raise Exception(f"Failed to fetch transcript: {str(e)}")
            
            # Display transcript
            with st.expander("View Transcript", expanded=True):
                st.text_area("Transcript", transcript_text, height=250, label_visibility="collapsed")
            
            # Add download button
            st.download_button(
                label="Download Transcript",
                data=transcript_text,
                file_name=f"transcript_{video_id}.txt",
                mime="text/plain"
            )
            
            # Summarize if requested
            if summarize and transcript_text.strip():
                status_text.text("🧠 Processing transcript with AI...")
                progress_bar.progress(60)
                
                cleaned = clean_transcript(transcript_text)
                if not cleaned:
                    st.warning("The transcript is empty after cleaning. Cannot generate summary.")
                else:
                    prompt_prefix = (
                        "Please provide a concise summary of the following text, "
                        "focusing on the main points and key information. "
                        "Keep it well-structured and easy to read.\n\n"
                    )
                    
                    summarizer = load_summarizer()
                    chunks = chunk_text(cleaned, max_len=1500)
                    
                    with st.spinner(f"Generating summary (processing {len(chunks)} chunks)..."):
                        summaries = []
                        for i, chunk in enumerate(chunks, 1):
                            status_text.text(f"📝 Processing chunk {i} of {len(chunks)}...")
                            progress = 60 + int(30 * (i / len(chunks)))
                            progress_bar.progress(progress)
                            
                            summary = summarizer(
                                prompt_prefix + chunk,
                                max_length=250,
                                min_length=50,
                                do_sample=False,
                                temperature=0.7,
                                repetition_penalty=1.2,
                                top_p=0.95
                            )[0]['summary_text']
                            summaries.append(summary)
                        
                        # If multiple chunks, merge summaries
                        if len(summaries) > 1:
                            status_text.text("🔗 Combining summaries...")
                            progress_bar.progress(95)
                            combined_summary = " ".join(summaries)
                            final_summary = summarizer(
                                prompt_prefix + combined_summary,
                                max_length=300,
                                min_length=100,
                                do_sample=False
                            )[0]['summary_text']
                        else:
                            final_summary = summaries[0]
                        
                        # Display final summary
                        st.markdown("---")
                        st.subheader("📝 AI-Generated Summary")
                        with st.container():
                            st.markdown(f"""
                            <div style="
                                background-color: #233863;
                                padding: 1.5rem;
                                border-radius: 0.5rem;
                                margin: 1rem 0;
                            ">
                                {final_summary}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add download button for summary
                        st.download_button(
                            label="Download Summary",
                            data=final_summary,
                            file_name=f"summary_{video_id}.txt",
                            mime="text/plain"
                        )
        
        except NoTranscriptFound:
            st.error("❌ No English transcript available for this video.")
        except TranscriptsDisabled:
            st.error("❌ Transcripts are disabled for this video.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            
        progress_bar.progress(100)
        status_text.text("✅ Done!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
            
    except ValueError as ve:
        st.error(f"❌ {str(ve)}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        
elif not youtube_url and submitted:
    st.warning("⚠️ Please enter a YouTube URL.")
else:
    st.info("👆 Enter a YouTube URL and click 'Process Video' to get started.")

# Add footer
st.markdown("---")