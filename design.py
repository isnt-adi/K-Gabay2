"""
Design and styling for K-Gabay
Makes the app look nice and professional
"""

import streamlit as st

def apply_custom_styles():
    """
    Apply custom CSS styling to make K-Gabay look good
    This makes the sidebar look professional and chat messages clear
    """
    st.markdown("""
    <style>
        /* Make the sidebar look professional with dark blue theme */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #001122 0%, #002244 50%, #003366 100%) !important;
            padding: 1.5rem 1rem !important;
            border-right: 3px solid #003366 !important;
        }
        
        /* Make sidebar text white and readable */
        [data-testid="stSidebar"] * {
            color: white !important;
            font-weight: 500 !important;
        }
        
        /* Style sidebar headers */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
        }
        
        /* Style file upload area in sidebar */
        [data-testid="stSidebar"] .stFileUploader {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            padding: 15px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            margin-bottom: 20px !important;
        }
        
        /* Style upload buttons */
        [data-testid="stSidebar"] .stFileUploader button {
            color: #ffffff !important;
            background: linear-gradient(45deg, #0066cc, #003366) !important;
            border: 2px solid #ffffff !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stSidebar"] .stFileUploader button:hover {
            background: linear-gradient(45deg, #0080ff, #004488) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
        }
        
        /* Style text inputs in sidebar */
        [data-testid="stSidebar"] input {
            color: #ffffff !important;
            background: linear-gradient(135deg, #001122, #002244) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 8px !important;
            padding: 10px !important;
            font-weight: 500 !important;
        }
        
        [data-testid="stSidebar"] input:focus {
            border-color: #66ccff !important;
            box-shadow: 0 0 10px rgba(102, 204, 255, 0.3) !important;
        }
        
        [data-testid="stSidebar"] input::placeholder {
            color: rgba(255, 255, 255, 0.7) !important;
        }
        
        /* Style FAQ expandable sections */
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05)) !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            font-weight: 600 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
            margin-bottom: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.15)) !important;
        }
        
        [data-testid="stSidebar"] .streamlit-expanderContent {
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.1)) !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 15px !important;
            margin-bottom: 10px !important;
        }
        
        /* Main app background - light blue theme */
        [data-testid="stAppViewContainer"] {
            background-color: #e6f0ff !important;
        }
        
        .main .block-container {
            background-color: #e6f0ff !important;
        }
        
        /* Chat message styling - make user and assistant messages distinct */
        [data-testid="stChatMessage"] {
            padding: 15px !important;
            border-radius: 10px !important; 
            margin-bottom: 10px !important;
        }
        
        /* User messages - white background with blue border */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background-color: #ffffff !important;
            border: 2px solid #003366 !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * {
            color: #000000 !important;
        }
        
        /* Assistant messages - dark blue background with white text */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background-color: #003366 !important;
            border: none !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) * {
            color: white !important;
        }
        
        /* Style chat avatars */
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            background-color: #003366 !important;
            color: white !important;
            border: 2px solid #003366 !important;
        }
        
        [data-testid="stChatMessageAvatarUser"] *,
        [data-testid="stChatMessageAvatarAssistant"] * {
            color: white !important;
        }
        
        /* Style sources expander in assistant messages */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .streamlit-expanderHeader {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 5px !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .streamlit-expanderContent {
            background-color: rgba(0, 0, 0, 0.2) !important;
            border-radius: 5px !important;
            padding: 10px !important;
            color: white !important;
        }
        
        /* Chat input styling */
        [data-testid="stChatInput"] {
            background-color: white !important;
        }
        
        [data-testid="stChatInput"] input {
            background-color: white !important;
            color: #000000 !important;
            border: 1px solid #003366 !important;
        }
        
        /* Main area expanders (for audio/image upload) */
        .main .streamlit-expanderHeader {
            background-color: rgba(0, 51, 102, 0.1) !important;
            border-radius: 5px !important;
            color: #003366 !important;
        }
        
        .main .streamlit-expanderContent {
            background-color: rgba(230, 240, 255, 0.5) !important;
            border-radius: 5px !important;
            padding: 10px !important;
        }
        
        /* File uploaders in main area */
        .stFileUploader button {
            background-color: #003366 !important;
            color: white !important;
            border: 1px solid #003366 !important;
        }
        
        /* Status messages styling */
        [data-testid="stAlert"] div[data-baseweb="notification"] {
            background-color: rgba(51, 153, 255, 0.1) !important;
            border: 1px solid #3399ff !important;
            border-radius: 5px !important;
        }
        
        /* Error messages */
        .stError div[data-baseweb="notification"] {
            background-color: rgba(255, 59, 48, 0.1) !important;
            border: 1px solid #ff3b30 !important;
            color: #d70015 !important;
            border-radius: 5px !important;
        }
        
        /* Warning messages */
        .stWarning div[data-baseweb="notification"] {
            background-color: rgba(255, 193, 7, 0.1) !important;
            border: 1px solid #ffc107 !important;
            color: #856404 !important;
            border-radius: 5px !important;
        }
        
        /* Success messages */
        .stSuccess div[data-baseweb="notification"] {
            background-color: rgba(40, 167, 69, 0.1) !important;
            border: 1px solid #28a745 !important;
            color: #155724 !important;
            border-radius: 5px !important;
        }
        
        /* Loading spinner */
        .stSpinner > div {
            border-top-color: #003366 !important;
        }
        
        /* Main headers and titles */
        h1, h2, h3 {
            color: #003366 !important;
        }
        
        /* Caption text */
        .caption {
            color: #666666 !important;
        }
        
        /* Make section headers in sidebar more prominent */
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #66ccff !important;
            font-size: 1.1rem !important;
            margin-bottom: 10px !important;
        }
        
    </style>
    """, unsafe_allow_html=True)

def show_welcome_message():
    """
    Display a nice welcome message when the app starts
    """
    st.markdown("""
    ## Welcome to K-Gabay! 🤖
    
    Your AI-powered document assistant is ready to help you learn and understand your materials better.
    
    **Here's what I can do:**
    - Answer questions about uploaded PDF documents
    - Help with general academic questions
    - Support multiple languages
    - Process audio and image inputs
    
    **To get started:**
    1. Upload a PDF document in the sidebar, or
    2. Enter a webpage URL, or 
    3. Just ask me any academic question!
    """)

def show_processing_status(message="Processing..."):
    """
    Show a nice processing status message
    """
    return st.spinner(f"🔄 {message}")

def display_error_message(error_type, details=None):
    """
    Display user-friendly error messages with appropriate styling
    """
    error_messages = {
        "file_error": "📄 File Error",
        "processing_error": "⚙️ Processing Error", 
        "network_error": "🌐 Connection Error",
        "general_error": "❌ Error"
    }
    
    title = error_messages.get(error_type, error_messages["general_error"])
    
    if details:
        st.error(f"{title}: {details}")
    else:
        st.error(f"{title}: Something went wrong. Please try again.")

def display_success_message(message):
    """
    Display success messages with nice styling
    """
    st.success(f"✅ {message}")

def display_info_message(message):
    """
    Display info messages with nice styling
    """
    st.info(f"ℹ️ {message}")

def display_warning_message(message):
    """
    Display warning messages with nice styling
    """
    st.warning(f"⚠️ {message}")

# Functions available for import
__all__ = [
    'apply_custom_styles',
    'show_welcome_message',
    'show_processing_status',
    'display_error_message',
    'display_success_message', 
    'display_info_message',
    'display_warning_message'
]
