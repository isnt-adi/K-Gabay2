import streamlit as st
from backend.utils import translate_input, translate_output, transcribe_audio, extract_text_from_image, get_faqs
from backend.rag import initialize_qa_chain, initialize_general_qa
from design import apply_custom_styles
import os

# Basic app setup - make it look nice
st.set_page_config(page_title="K-Gabay", layout="wide")
apply_custom_styles()

# Show logo or title at the top
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists("logo.jfif"):
        st.image("logo.jfif", use_column_width=True)
    else:
        st.title("🤖 K-Gabay")
        st.caption("Your AI-Powered Document Assistant")

# Sidebar for file uploads and FAQs - keep it simple
with st.sidebar:
    st.subheader("📄 Upload a PDF")
    uploaded_file = st.file_uploader("Upload an educational PDF", type="pdf")
    
    st.subheader("🌐 Or Enter Web URL")
    web_url = st.text_input("Enter a webpage URL (optional)")
    
    # Show some helpful FAQs
    st.subheader("📌 FAQs")
    for faq in get_faqs():
        with st.expander(faq["question"]):
            st.write(faq["answer"])

# Function to check if we should reset chat when source changes
def should_reset_chat(new_source):
    """Simple function to check if we need to start fresh"""
    current_source = st.session_state.get("current_source", None)
    return current_source != new_source

# Determine what document to use - prioritize user uploads
current_document = None
source_name = None

if uploaded_file:
    # User uploaded a PDF
    try:
        current_document = uploaded_file.getvalue()  # Get the file content
        source_name = uploaded_file.name
        
        # Basic validation - is it actually a PDF?
        if len(current_document) < 100:
            st.error("❌ File seems empty or corrupted")
            current_document = None
        elif not current_document.startswith(b'%PDF'):
            st.error("❌ This doesn't look like a valid PDF file")
            current_document = None
        else:
            st.success(f"✅ PDF loaded: {uploaded_file.name}")
            
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        current_document = None
        
elif web_url and web_url.strip():
    # User entered a URL
    if web_url.startswith(('http://', 'https://')):
        current_document = web_url.strip()
        source_name = web_url
        st.success(f"✅ URL ready: {web_url}")
    else:
        st.error("❌ Please enter a valid URL (must start with http:// or https://)")
        current_document = None
else:
    # Try to use default PDF if available
    try:
        with open("backend/data/base.pdf", "rb") as f:
            current_document = f.read()
        source_name = "default_pdf"
        st.info("📄 Using default PDF. Upload your own for specific answers!")
    except FileNotFoundError:
        st.warning("⚠️ No document loaded. You can still ask general questions!")
        current_document = None

# Initialize our AI systems based on what we have
if current_document:
    # We have a document - set up document-based QA
    if should_reset_chat(source_name):
        with st.spinner("Processing document..."):
            try:
                # Initialize the QA system with our document
                qa_system = initialize_qa_chain(current_document, 
                                              "url" if isinstance(current_document, str) else "pdf")
                
                # Store everything in session state for later use
                st.session_state.document_qa = qa_system["qa_function"]
                st.session_state.vectorstore = qa_system["vectorstore"] 
                st.session_state.general_qa = qa_system["general_qa_function"]
                st.session_state.current_source = source_name
                
                # Reset chat when we get a new document
                st.session_state.messages = [
                    {"role": "assistant", 
                     "content": "Hi! I've processed your document and I'm ready to answer questions about it. I can also help with general academic questions!"}
                ]
                
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                # Fall back to general QA only
                st.session_state.document_qa = None
                st.session_state.vectorstore = None
                st.session_state.general_qa = initialize_general_qa()
else:
    # No document - just general QA
    if "general_qa" not in st.session_state:
        st.session_state.general_qa = initialize_general_qa()
        st.session_state.document_qa = None
        st.session_state.vectorstore = None

# Set up default chat messages if we don't have any
if "messages" not in st.session_state:
    if st.session_state.get("document_qa"):
        welcome_msg = "Hi! I've got your document ready. Ask me anything about it, or any general academic question!"
    else:
        welcome_msg = "Hi! I'm K-Gabay. Upload a document for specific answers, or ask me general academic questions!"
    
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

# Display all previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if this is an assistant message with sources
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources from Document"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"**Source {i+1}:**")
                    st.write(source)
                    if i < len(message["sources"]) - 1:
                        st.divider()

# Get user input - text, audio, or image
user_input = st.chat_input("Ask me anything!")

# Audio and image upload section - collapsible to save space
with st.expander("🎧 Upload Audio or Image", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        # Audio upload and transcription
        audio_file = st.file_uploader("🎙️ Upload audio", type=["wav", "mp3"], key="audio")
        if audio_file:
            with st.spinner("Converting speech to text..."):
                transcribed = transcribe_audio(audio_file)
            if transcribed:
                user_input = transcribed
                st.success(f"Got it: {user_input}")
            else:
                st.warning("Couldn't understand the audio")

    with col2:
        # Image upload and text extraction
        image_file = st.file_uploader("🖼️ Upload image", type=["png", "jpg", "jpeg"], key="image")
        if image_file:
            with st.spinner("Reading text from image..."):
                extracted = extract_text_from_image(image_file)
            if extracted:
                user_input = extracted
                st.success(f"Found text: {user_input}")
            else:
                st.warning("No text found in image")

# Handle user input when they ask something
if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Translate input if needed (for non-English users)
    translated_question, detected_language = translate_input(user_input)

    # Generate response
    with st.spinner("🤔 Thinking..."):
        try:
            response_data = {"role": "assistant", "content": ""}
            
            # Try document-specific answer first if we have a document
            if st.session_state.get("document_qa") and st.session_state.get("vectorstore"):
                
                # Check if question is relevant to the document
                # Do a quick search to see if we find related content
                similar_docs = st.session_state.vectorstore.similarity_search(translated_question, k=3)
                
                # Simple relevance check - count matching words
                question_words = set(translated_question.lower().split())
                relevance_score = 0
                
                for doc in similar_docs:
                    doc_words = set(doc.page_content.lower().split())
                    common_words = question_words.intersection(doc_words)
                    relevance_score += len(common_words)
                
                # If we found enough matching content, use document QA
                if relevance_score > 2:  # At least 3 matching words
                    try:
                        # Get answer from document
                        answer = st.session_state.document_qa(translated_question)
                        final_answer = translate_output(answer, detected_language)
                        
                        # Get sources to show user where info came from
                        sources = []
                        for doc in similar_docs:
                            content = doc.page_content.strip()
                            if len(content) > 50:  # Only include substantial content
                                sources.append(content)
                        
                        response_data = {
                            "role": "assistant",
                            "content": final_answer,
                            "sources": sources[:3]  # Limit to top 3 sources
                        }
                        
                    except Exception as e:
                        # Document QA failed, fall back to general
                        if st.session_state.get("general_qa"):
                            answer = st.session_state.general_qa(translated_question)
                            final_answer = translate_output(answer, detected_language)
                            response_data["content"] = final_answer + "\n\n*Note: Answered with general knowledge (document search had issues)*"
                        else:
                            response_data["content"] = f"❌ Sorry, encountered an error: {str(e)}"
                else:
                    # Question not relevant to document, use general QA
                    if st.session_state.get("general_qa"):
                        answer = st.session_state.general_qa(translated_question)
                        final_answer = translate_output(answer, detected_language)
                        response_data["content"] = final_answer + "\n\n*Note: This is general knowledge (not from your document)*"
                    else:
                        response_data["content"] = "I can answer document questions, but need setup for general questions."
            else:
                # No document available - general QA only
                if st.session_state.get("general_qa"):
                    answer = st.session_state.general_qa(translated_question)
                    final_answer = translate_output(answer, detected_language)
                    response_data["content"] = final_answer
                else:
                    response_data["content"] = "❌ Please upload a document or wait for the system to initialize."
                
        except Exception as e:
            response_data = {
                "role": "assistant",
                "content": f"❌ Sorry, something went wrong: {str(e)}"
            }

    # Add response to chat history
    st.session_state.messages.append(response_data)
    
    # Show the response
    with st.chat_message("assistant"):
        st.markdown(response_data["content"])
        
        # Show sources if available
        if "sources" in response_data and response_data["sources"]:
            with st.expander("📚 Sources from Document"):
                for i, source in enumerate(response_data["sources"]):
                    st.write(f"**Source {i+1}:**")
                    st.write(source)
                    if i < len(response_data["sources"]) - 1:
                        st.divider()
