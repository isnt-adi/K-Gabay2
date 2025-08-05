"""
Utility functions for K-Gabay
Handles translation, audio transcription, image text extraction, and FAQs
"""

import streamlit as st
from deep_translator import GoogleTranslator
import speech_recognition as sr
from PIL import Image
import pytesseract
import io
import logging

# Set up logging
logger = logging.getLogger(__name__)

def translate_input(text):
    """
    Translate user input to English if needed
    Returns: (translated_text, detected_language)
    """
    try:
        # Translate from any language to English
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        logger.info("Translation to English successful")
        return translated_text, 'auto'
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text, 'en'

def translate_output(text, target_language):
    """
    Translate AI response back to user's language if needed
    """
    if target_language == 'en':
        return text
    try:
        translated = GoogleTranslator(source='en', target=target_language).translate(text)
        return translated
    except Exception as e:
        logger.warning(f"Output translation failed: {e}")
        return text

def transcribe_audio(audio_file):
    """
    Convert audio file to text using speech recognition
    """
    try:
        recognizer = sr.Recognizer()
        audio_data = audio_file.read()
        audio_file_like = io.BytesIO(audio_data)
        with sr.AudioFile(audio_file_like) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            logger.info("Audio transcription successful")
            return text
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return None

def extract_text_from_image(image_file):
    """
    Extract text from image using OCR (Optical Character Recognition)
    """
    try:
        image = Image.open(image_file)
        extracted_text = pytesseract.image_to_string(image)
        cleaned_text = extracted_text.strip()
        if cleaned_text:
            logger.info("Text extraction from image successful")
            return cleaned_text
        else:
            logger.warning("No text found in image")
            return None
    except Exception as e:
        logger.error(f"Image text extraction failed: {e}")
        return None

def get_faqs():
    """
    Return a list of frequently asked questions for the sidebar
    """
    faqs = [
        {
            "question": "What types of documents can I upload?",
            "answer": "You can upload PDF files containing educational content like textbooks, research papers, course materials, or study guides. I work best with text-based PDFs."
        },
        {
            "question": "Can I ask questions without uploading documents?",
            "answer": "Yes! I can answer general academic questions about various subjects like math, science, history, and more using my built-in knowledge."
        },
        {
            "question": "What languages do you support?",
            "answer": "I can understand and respond in multiple languages. Ask your question in your preferred language and I'll detect it automatically."
        },
        {
            "question": "How do I upload audio or images?",
            "answer": "Click on the 'Upload Audio or Image' section below the chat. I can transcribe speech from audio files or extract text from images with text in them."
        },
        {
            "question": "Can you help with homework and assignments?",
            "answer": "I can help explain concepts, provide study guidance, and answer questions about your materials. However, I encourage learning and understanding rather than just providing direct answers."
        },
        {
            "question": "What if my document is very long?",
            "answer": "I can handle long documents! I break them into smaller sections and search through all of them to find relevant information for your questions."
        }
    ]
    return faqs

def validate_file_upload(uploaded_file, file_type="pdf"):
    """
    Basic validation for uploaded files
    Returns: (is_valid, error_message)
    """
    try:
        if uploaded_file is None:
            return False, "No file uploaded"
        file_size = len(uploaded_file.getvalue())
        if file_size > 10 * 1024 * 1024:
            return False, "File is too large (max 10MB)"
        if file_size < 100:
            return False, "File appears to be empty"
        if file_type == "pdf":
            file_content = uploaded_file.getvalue()
            if not file_content.startswith(b'%PDF'):
                return False, "File doesn't appear to be a valid PDF"
        return True, "File is valid"
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

def format_sources_for_display(sources, max_sources=3):
    """
    Format source content for nice display in the UI
    """
    if not sources:
        return []
    formatted_sources = []
    for i, source in enumerate(sources[:max_sources]):
        cleaned_source = source.strip()
        if len(cleaned_source) > 500:
            cleaned_source = cleaned_source[:500] + "..."
        cleaned_source = ' '.join(cleaned_source.split())
        if cleaned_source and len(cleaned_source) > 50:
            formatted_sources.append(cleaned_source)
    return formatted_sources

def create_simple_error_message(error_type, context=""):
    """
    Create user-friendly error messages instead of technical ones
    """
    error_messages = {
        "file_read": "I had trouble reading your file. Please make sure it's a valid PDF and try uploading again.",
        "processing": "I encountered an issue while processing your document. The file might be corrupted or in an unsupported format.",
        "network": "I'm having trouble connecting to external services. Please check your internet connection and try again.",
        "model_load": "I'm having trouble loading the AI models. This might be a temporary issue - please try again in a moment.",
        "translation": "I had trouble with language translation. Your question will be processed in English.",
        "audio": "I couldn't process the audio file. Please make sure it's a clear recording in WAV or MP3 format.",
        "image": "I couldn't extract text from the image. Please make sure the image contains clear, readable text.",
        "general": "Something went wrong, but don't worry! Please try again or rephrase your question."
    }
    base_message = error_messages.get(error_type, error_messages["general"])
    return f"{base_message} Additional info: {context}" if context else base_message

__all__ = [
    'translate_input',
    'translate_output', 
    'transcribe_audio',
    'extract_text_from_image',
    'get_faqs',
    'validate_file_upload',
    'format_sources_for_display',
    'create_simple_error_message'
]
