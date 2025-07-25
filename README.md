# 🎬 YouTube Transcript Summarizer

A Streamlit application that extracts transcripts from YouTube videos and generates AI-powered summaries using the Transformers library.

## ✨ Features

- Extract transcripts from any YouTube video with available captions
- Generate concise AI-powered summaries of video content
- Clean and preprocess transcripts for better readability
- Download transcripts and summaries as text files
- Responsive and user-friendly interface

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd yt-transcript-summarizer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App Locally

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

## 🛠 Usage

1. Enter a YouTube video URL in the input field
2. Toggle the "Generate AI Summary" option if you want an AI-generated summary
3. Click "Process Video" to fetch and process the transcript
4. View the transcript and summary in the app
5. Download the transcript or summary using the download buttons

## 🌐 Deployment

### Deploying to Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and select your repository
4. Set the branch and main file path (`app.py`)
5. Click "Deploy"

### Deploying with Docker

1. Build the Docker image:
   ```bash
   docker build -t yt-transcript-summarizer .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 yt-transcript-summarizer
   ```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or feedback, please open an issue on GitHub.
