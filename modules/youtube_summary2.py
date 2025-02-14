# import re
# from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api._errors import VideoUnavailable, TranscriptsDisabled, NoTranscriptFound
# from transformers import pipeline

# # Load Hugging Face summarizer pipeline
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# # Function to extract video ID from a YouTube URL
# def get_video_id(url):
#     match = re.search(r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|(?:v|e(?:mbed)?)%2F|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})", url)
#     return match.group(1) if match else None

# # Fetch transcript using youtube_transcript_api
# def fetch_transcript(video_id):
#     try:
#         transcript = YouTubeTranscriptApi.get_transcript(video_id)
#         return " ".join([entry['text'] for entry in transcript])
#     except (VideoUnavailable, TranscriptsDisabled, NoTranscriptFound):
#         return None

# # Function to summarize transcript text
# def summarize_text(text):
#     if not text:
#         return "No transcript available to summarize."

#     chunk_size = 2000  # Adjust based on model limitations
#     text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
#     summaries = []
#     for chunk in text_chunks:
#         if chunk.strip():  # Ensure non-empty chunks
#             try:
#                 summary = summarizer(chunk, max_length=min(1800, len(chunk)), min_length=30, do_sample=False)[0]['summary_text']
#                 summaries.append(summary)
#             except Exception as e:
#                 summaries.append(f"Error summarizing chunk: {e}")

#     return " ".join(summaries) if summaries else "No summary generated."

# # Main function to summarize a YouTube video
# def youtube_summary_app(video_url):
#     video_id = get_video_id(video_url)
#     if not video_id:
#         return "Invalid YouTube URL. Please provide a valid link."

#     transcript = fetch_transcript(video_id)
#     if not transcript:
#         return "No transcript available for this video."

#     summary = summarize_text(transcript)
#     return f"**Summary:** {summary}"


import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import VideoUnavailable, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline

# Load Hugging Face summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to extract video ID from a YouTube URL
def get_video_id(url):
    match = re.search(r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|(?:v|e(?:mbed)?)%2F|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# Fetch transcript using youtube_transcript_api
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except (VideoUnavailable, TranscriptsDisabled, NoTranscriptFound):
        return None

# Function to summarize transcript text
def summarize_text(text):
    if not text:
        return "No transcript available to summarize."

    chunk_size = 2000  # Adjust based on model limitations
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    summaries = []
    for chunk in text_chunks:
        if chunk.strip():  # Ensure non-empty chunks
            try:
                summary = summarizer(chunk, max_length=min(1800, len(chunk)), min_length=30, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                summaries.append(f"Error summarizing chunk: {e}")

    return summaries if summaries else ["No summary generated."]

# Function to format summary into bullet points
def format_as_bullets(summaries):
    return "\n".join([f"- {point}" for point in summaries])

# Main function to summarize a YouTube video
def youtube_summary_app(video_url):
    video_id = get_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL. Please provide a valid link."

    transcript = fetch_transcript(video_id)
    if not transcript:
        return "No transcript available for this video."

    summaries = summarize_text(transcript)
    bullet_points = format_as_bullets(summaries)
    return f"**Summary:**\n{bullet_points}"

# Example usage
# video_url = "https://www.youtube.com/watch?v=example"
# print(youtube_summary_app(video_url))
