# 🎓 K-Gabay: AI-Powered Academic Assistant

**K-Gabay** is a multilingual AI assistant designed to help students with academic research and document analysis. Powered by Retrieval-Augmented Generation (RAG) technology, it provides intelligent answers from uploaded documents while supporting general academic queries.

🔗 Live Demo (Coming soon)

## 🌟 Key Features

| Feature             | Description                                               |
|---------------------|-----------------------------------------------------------|
| 📂 Document Intelligence | Upload PDFs or enter URLs for document-specific answers     |
| 🧠 Hybrid Knowledge      | Combines document insights with general academic knowledge |
| 🌍 Multilingual Support  | Understands and responds in multiple languages             |
| 🖼️ Image Processing      | Extract and analyze text from images                        |
| 🎙️ Audio Input           | Ask questions via voice recordings                          |
| 📚 Source Citation       | Shows document sources for answers                          |
| 💬 Contextual Chat       | Maintains conversation context                              |

## 🛠️ Technology Stack

### Core Components
- **Frontend**: Streamlit
- **Backend**: Python
- **AI Models**:
  - Microsoft DialoGPT (primary)
  - Mistral-7B (fallback)
- **RAG Engine**: LangChain + FAISS
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Speech-to-Text**: SpeechRecognition
- **OCR**: pytesseract
- **Translation**: googletrans

### Development Tools
- **Vibecoding**: Used for rapid prototyping and iterative development
- **AI Assistants**: Claude, DeepSeek, Gemini, and ChatGPT contributed to code refinement and troubleshooting

## 🚀 Installation Guide

### Prerequisites
- Python 3.10+
- Tesseract OCR (for image processing)

### Setup
```
# Clone repository
git clone https://github.com/your-repo/K-Gabay.git
cd K-Gabay

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

🏃 Running the Application
```
streamlit run app.py
```

📖 Usage Examples
Document Analysis:
Upload a research paper and ask specific questions like:

"Summarize the key findings from this paper"

General Knowledge:
"Explain quantum computing basics"

"What are the main causes of World War I?"

Multimedia Input:
Upload a photo of textbook pages

Record a voice question about the material

🏗️ Project Structure
```
K-Gabay/
├── app.py                  # Main application
├── design.py               # UI styling and components
├── requirements.txt        # Dependencies
├── logo.jfif               # Application logo
│
├── backend/
│   ├── rag.py              # RAG implementation
│   ├── utils.py            # Utility functions
│   └── syst_instructions.py # AI prompt engineering
│
└── README.md               # This documentation
```

🤝 Acknowledgments
This project was developed with significant assistance from:

Vibecoding methodology for efficient development cycles

AI Assistants: Claude, DeepSeek, Gemini, and ChatGPT for:

Code optimization suggestions

Debugging assistance

Architectural advice

Documentation support

📜 License
MIT License - Available for educational and non-commercial use

🌐 SDG Alignment
Supports UN Sustainable Development Goal 4 (Quality Education) by making academic resources more accessible through AI assistance.

💡 Tip: For best results, upload well-structured PDFs with clear text. The system works best with academic materials like research papers, textbooks, and educational resources.
