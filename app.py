import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
import time
from typing import List, Optional, Dict, Any

# Optional imports for summarization
SUMMARIZATION_AVAILABLE = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    SUMMARIZATION_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Summarization features are disabled because required packages are not installed.")
    st.warning("To enable summarization, install the required packages with: `pip install transformers torch sentencepiece`")

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

def summarize_text(text: str, model_name: str = "facebook/bart-large-cnn") -> Optional[str]:
    """Summarize the given text using a pre-trained model if available."""
    if not SUMMARIZATION_AVAILABLE:
        return None
        
    try:
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Initialize the summarization pipeline
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        
        # Split the text into chunks that fit within the model's maximum length
        max_chunk_length = 1024  # Adjust based on the model's max length
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        # Generate summary for each chunk and combine
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)
    except Exception as e:
        st.warning(f"⚠️ Summarization failed: {str(e)}")
        return None

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
    
    st.header("Settings")
    enable_summary = st.checkbox(
        "Enable AI Summary", 
        value=SUMMARIZATION_AVAILABLE, 
        disabled=not SUMMARIZATION_AVAILABLE,
        help="AI-powered summarization is not available. Install required packages to enable." if not SUMMARIZATION_AVAILABLE else "Enable AI-powered summarization (requires more resources)"
    )

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
            if enable_summary and SUMMARIZATION_AVAILABLE:
                with st.spinner("Generating summary (this may take a minute)..."):
                    summary = summarize_text(transcript_text)
                    if summary:
                        st.session_state.summary = summary
                        st.success("Summary generated successfully!")
                    else:
                        st.error("Failed to generate summary. The transcript may be too short or there was an error with the summarization model.")
                        st.session_state.summary = None
            elif enable_summary and not SUMMARIZATION_AVAILABLE:
                st.warning("Summarization is not available. Please install the required packages to enable this feature.")
                st.code("pip install transformers torch sentencepiece")
            
            if 'summary' in st.session_state:
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
                        {st.session_state.summary}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add download button for summary
                st.download_button(
                    label="Download Summary",
                    data=st.session_state.summary,
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