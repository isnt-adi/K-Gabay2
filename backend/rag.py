from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import torch
import logging
import re

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store our AI models (so we don't reload them every time)
tokenizer = None
model = None
general_pipeline = None

def initialize_models():
    """
    Load AI models for text generation - try simple models first
    Returns the tokenizer and model if successful
    """
    global tokenizer, model, general_pipeline
    
    # If we already loaded models, don't do it again
    if tokenizer is not None and model is not None:
        return tokenizer, model
    
    try:
        # Check if we have a GPU or just CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Try different models from simple to complex
        model_options = [
            "microsoft/DialoGPT-medium",  # This usually works
            "microsoft/DialoGPT-small",   # Backup option
        ]
        
        for model_name in model_options:
            try:
                logger.info(f"Trying to load {model_name}")
                
                # Load the tokenizer (converts text to numbers)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Add padding token if missing
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load the actual model
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Move to correct device
                model = model.to(device)
                
                logger.info(f"Successfully loaded {model_name}")
                
                # Create a pipeline for easy text generation
                try:
                    general_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if device == "cuda" else -1,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32
                    )
                    logger.info("Text generation pipeline ready")
                except Exception as e:
                    logger.warning(f"Pipeline creation failed: {e}")
                
                return tokenizer, model
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        logger.error("All model loading attempts failed")
        return None, None
        
    except Exception as e:
        logger.error(f"Model initialization completely failed: {e}")
        return None, None

def initialize_general_qa():
    """
    Set up a function that can answer general academic questions
    """
    global general_pipeline

    # Try to load models if we haven't already
    if general_pipeline is None:
        initialize_models()

    def answer_general_question(question):
        """
        Answer general questions using AI model or simple fallbacks
        """
        try:
            # Handle basic math questions directly
            if re.search(r'(\bwhat is\b|\bcalculate\b|\bsolve\b).*\d+\s*[\+\-\*\/]\s*\d+', question.lower()):
                try:
                    # Find and calculate simple math expressions
                    expr = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', question)
                    if expr:
                        result = eval(expr.group(0))  # This is safe for basic math
                        return f"The answer is {result}"
                except:
                    pass

            # Check if it's an academic question
            academic_keywords = [
                'explain', 'what is', 'how does', 'why', 'define', 'describe',
                'analyze', 'compare', 'discuss', 'calculate', 'solve', 'theory',
                'concept', 'principle', 'formula', 'equation', 'law', 'university', 
                'college', 'application', 'deadline', 'admission', 'requirement',
                'math', 'science', 'history', 'biology', 'chemistry', 'physics'
            ]

            question_lower = question.lower()
            is_academic = any(keyword in question_lower for keyword in academic_keywords)

            if is_academic:
                # Try to use our AI model if available
                if general_pipeline:
                    try:
                        prompt = f"Question: {question}\nAnswer: "
                        
                        response = general_pipeline(
                            prompt,
                            max_new_tokens=200,  # Keep responses reasonable length
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            repetition_penalty=1.1,
                            pad_token_id=general_pipeline.tokenizer.eos_token_id,
                            truncation=True
                        )
                        
                        generated_text = response[0]['generated_text']
                        
                        # Extract just the answer part
                        if "Answer:" in generated_text:
                            answer = generated_text.split("Answer:")[-1].strip()
                        else:
                            answer = generated_text.replace(prompt, "").strip()
                        
                        # Clean up the response
                        answer = clean_up_answer(answer)
                        
                        if len(answer) > 20:  # Make sure we got a decent answer
                            return answer
                    except Exception as e:
                        logger.error(f"AI generation failed: {e}")
                
                # Fallback to simple responses if AI fails
                return create_simple_response(question)
            else:
                return "I'm designed to help with academic and educational questions. Could you ask about a specific topic you're studying?"
            
        except Exception as e:
            logger.error(f"General QA failed: {e}")
            return "I encountered an error. Please try rephrasing your question or ask about a specific academic topic."
    
    return answer_general_question

def clean_up_answer(answer):
    """
    Clean up AI-generated answers to remove repetition and nonsense
    """
    # Split into lines and remove repetitive ones
    lines = answer.split('\n')
    unique_lines = []
    seen = set()
    
    for line in lines:
        line = line.strip()
        if line and line not in seen and len(line) > 5:
            unique_lines.append(line)
            seen.add(line)
    
    cleaned = ' '.join(unique_lines)
    
    # Remove excessive word repetition
    words = cleaned.split()
    if len(words) > 10:
        # Check for repetitive patterns
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears too much, it's probably repetitive garbage
        max_count = max(word_counts.values())
        if max_count > len(words) * 0.3:  # More than 30% repetition
            cleaned = ' '.join(words[:len(words)//2])  # Take first half
    
    return cleaned

def create_simple_response(question):
    """
    Create simple template responses when AI models aren't working
    """
    question_lower = question.lower()
    
    # Pattern matching for common question types
    if any(word in question_lower for word in ['what is', 'define', 'definition']):
        return "This is asking for a definition. For detailed definitions of academic terms, I'd recommend uploading a relevant textbook or course material for more specific information."
    
    elif any(word in question_lower for word in ['how', 'process', 'steps']):
        return "This seems to be asking about a process or method. Academic processes can vary by field and context. Uploading relevant course materials would help me give more specific guidance."
    
    elif any(word in question_lower for word in ['why', 'reason', 'cause']):
        return "This is asking for an explanation of reasons or causes. For detailed academic explanations, having relevant source materials would help me provide more comprehensive answers."
    
    elif any(word in question_lower for word in ['calculate', 'solve', 'formula']):
        return "This appears to be a math or problem-solving question. For accurate calculations and step-by-step solutions, uploading your textbook or problem sets would be very helpful."
    
    else:
        return "I can help with academic questions when you provide more context or upload relevant educational materials. What subject are you studying?"

def create_document_qa_function(vectorstore):
    """
    Create a function that answers questions based on uploaded documents
    """
    def answer_from_document(question):
        """
        Search the document and create an answer from relevant parts
        """
        try:
            # Search for relevant parts of the document
            relevant_docs = vectorstore.similarity_search(question, k=5)  # Get top 5 matches
            
            if not relevant_docs:
                return "I couldn't find relevant information in the document to answer your question."
            
            # Combine the relevant content
            all_content = []
            for doc in relevant_docs:
                content = doc.page_content.strip()
                if content and len(content) > 30:  # Only include substantial content
                    all_content.append(content)
            
            if not all_content:
                return "I couldn't find specific information to answer your question in the document."
            
            # Join the content together
            combined_content = " ".join(all_content)
            
            # Create a comprehensive answer by finding relevant sentences
            answer = create_answer_from_content(combined_content, question)
            
            return answer
            
        except Exception as e:
            logger.error(f"Document QA failed: {e}")
            return f"I encountered an error while searching the document: {str(e)}"
    
    return answer_from_document

def create_answer_from_content(content, question):
    """
    Extract and organize relevant information from document content to answer the question
    """
    # Find keywords from the question
    question_words = re.findall(r'\b\w+\b', question.lower())
    important_words = [w for w in question_words if len(w) > 3]  # Focus on longer words
    
    # Split content into sentences
    sentences = re.split(r'[.!?]+', content)
    
    # Score sentences based on how many question words they contain
    scored_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Only consider substantial sentences
            sentence_lower = sentence.lower()
            score = sum(1 for word in important_words if word in sentence_lower)
            
            if score > 0:
                scored_sentences.append((sentence, score))
    
    # Sort by score and take the best ones
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in scored_sentences[:5]]  # Take top 5 sentences
    
    if not top_sentences:
        return "I found some content in the document, but it doesn't seem directly related to your question."
    
    # Combine sentences into a flowing answer
    if len(top_sentences) == 1:
        return top_sentences[0]
    elif len(top_sentences) == 2:
        return f"{top_sentences[0]} Additionally, {top_sentences[1].lower()}"
    else:
        answer = top_sentences[0]
        connectors = ["Furthermore,", "Additionally,", "It's also noted that", "The document also mentions that"]
        
        for i, sentence in enumerate(top_sentences[1:4]):  # Add up to 3 more sentences
            connector = connectors[i % len(connectors)]
            answer += f" {connector} {sentence.lower()}"
        
        return answer

def initialize_qa_chain(source_data, source_type="pdf"):
    """
    Main function to set up the question-answering system
    Takes either PDF data or URL and returns QA functions
    """
    try:
        logger.info(f"Initializing QA system for {source_type}")
        
        # Load documents based on type
        documents = []
        if source_type == "pdf":
            # Handle PDF files
            if not isinstance(source_data, bytes):
                raise ValueError("PDF data must be bytes")
            
            if len(source_data) < 100:
                raise ValueError("PDF file is too small")
            
            if not source_data.startswith(b'%PDF'):
                raise ValueError("Not a valid PDF file")
            
            # Create temporary file to process PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(source_data)
                tmp_file.flush()
                tmp_path = tmp_file.name
            
            try:
                # Load PDF content
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} pages from PDF")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        elif source_type == "url":
            # Handle web URLs
            try:
                loader = WebBaseLoader(source_data)
                documents = loader.load()
                logger.info(f"Loaded content from URL")
            except Exception as e:
                raise ValueError(f"Could not load webpage: {str(e)}")
        
        if not documents:
            raise ValueError("No content could be loaded")
        
        # Check if we got meaningful content
        total_content = sum(len(doc.page_content) for doc in documents)
        if total_content < 100:
            raise ValueError("Document appears to be empty or too short")
        
        # Split documents into smaller chunks for better searching
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Size of each chunk
            chunk_overlap=100,  # Overlap between chunks to maintain context
            separators=["\n\n", "\n", ". ", "! ", "? ", " "],  # Split on these
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        
        if not chunks:
            raise ValueError("Document splitting failed")
        
        # Create vector database for similarity search
        try:
            # Use embeddings to convert text to numbers for similarity search
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            # Create the vector database
            vectorstore = FAISS.from_documents(chunks, embeddings)
            logger.info("Vector database created successfully")
            
        except Exception as e:
            raise ValueError(f"Could not create search database: {str(e)}")
        
        # Create our QA functions
        document_qa_function = create_document_qa_function(vectorstore) 
        general_qa_function = initialize_general_qa()
        
        # Return everything the main app needs
        return {
            "qa_function": document_qa_function,
            "vectorstore": vectorstore, 
            "general_qa_function": general_qa_function
        }
        
    except Exception as e:
        logger.error(f"QA system initialization failed: {e}")
        # Even if document processing fails, try to provide general QA
        general_qa_function = initialize_general_qa()
        return {
            "qa_function": None,
            "vectorstore": None,
            "general_qa_function": general_qa_function
        }
