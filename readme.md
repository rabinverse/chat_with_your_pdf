# Chat with Your PDF

An Streamlit application that lets you upload multiple PDF documents and ask questions about them using AI-powered retrieval and generation.

## Features 

- **📤 Easy PDF Upload** — Simply upload any PDF and the app automatically extracts text and creates embeddings
- **💾 Persistent Storage** — Previously processed PDFs are cached, so you can re-ask questions without re-processing
- **Smart Retrieval** — Two-step retrieval system:
  - **Instant page numbers** — Fast similarity search shows relevant page numbers immediately.It finds the best page number in the book for situation with no internet
  - **Smart answers** — Full AI-generated answers appear while you wait
- **Multi-PDF Support** — Compare information across multiple uploaded PDFs
- **Manage PDFs** — Delete vector stores of PDFs you no longer need
- **Intelligent Extraction** — Handles multiple PDF types:
  - Digital PDFs (text-based)
  - Complex layouts (block detection)
  - Scanned PDFs (OCR with Tesseract)

## Project Architecture

```
├── src/                          # Main application code
│   ├── app.py                   # Streamlit UI entry point
│   ├── config.py                # Central configuration
│   ├── ingestion.py             # PDF extraction & chunking
│   ├── embedding_model.py       # Embedding model setup
│   ├── vectore_store.py         # Chroma vector store management
│   ├── llm.py                   # LLM provider setup
│   ├── retriever.py             # Two-step retrieval logic
│   └── prompt_design.py         # LLM prompt templates
├── notebooks/                    # Jupyter notebooks for exploration
├── usefulcomponents/            # Runtime artifacts
│   ├── embedding_cache/         # Downloaded embedding models
│   └── vectorstores/            # PDF vector databases (organized by PDF name)
├── requirements.txt             # Python dependencies
```

## Prerequisites

Before you start, make sure you have:

- **Python 3.9+** (3.9, 3.10, or 3.11 recommended)
- **pip** (Python package manager)
- **Tesseract OCR** (optional, for scanning PDF support)

### Install Tesseract (Optional for Scanned PDFs)

<details>
<summary><b>macOS</b></summary>

```bash
brew install tesseract
```

</details>

<details>
<summary><b>Ubuntu/Debian</b></summary>

```bash
sudo apt-get install tesseract-ocr
```

</details>

<details>
<summary><b>Windows</b></summary>

Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

</details>

### API Keys Required

You'll need API keys for the LLM providers you want to use:

- **Groq** (Free, fast): Get your key at https://console.groq.com/keys
- **Google Gemini** (Optional): Get your key at https://aistudio.google.com/app/api-keys

## Installation (Step-by-Step)

### Step 1: Clone or Navigate to Your Project

```bash
git clone https://github.com/rabinverse/chat_with_your_pdf.git
```

### Step 2: Create a Virtual Environment

A virtual environment keeps your project dependencies isolated from your system Python.

```bash
# Create a virtual environment
python3 -m venv chatwithpdfvenv

# Activate it
# On macOS/Linux
source chatwithpdfvenv/bin/activate
# OR
# On Windows
chatwithpdfvenv\Scripts\activate

#or virtual environment with conda ->recommended
conda create --name chatwithpdfvenv

#activate
conda activate chatwithpdfvenv
```

**You should see `(chatwithpdfvenv)` in your terminal prompt after activation.**

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This may take 2-5 minutes depending on your internet speed.**

### Step 4: Create a .env File

Create a .env file in your project root with your API keys:

```bash
# .env
GROQ_API_KEY="your_groq_api_key_here"
GOOGLE_API_KEY="your_google_gemini_api_key_here"
HF_TOKEN="your_huggingface_token"

```
visit
- **Groq** : Get your key at https://console.groq.com/keys
- **Google Gemini** (Optional): Get your key at https://aistudio.google.com/app/api-keys



### Step 5: Verify Installation

Test that everything is set up correctly:

```bash
python3 -c "import streamlit; import langchain; import chromadb; print(' All dependencies installed!')"
```

## Running the Application

### Start the Streamlit App

```bash
streamlit run src/app.py
```

**Open http://localhost:8501 in your browser.**

### Stop the App

Press `Ctrl+C` in your terminal to stop the server.

##

## How to Use

### Basic Workflow

1. **Upload a PDF**
   - Click " Upload a PDF" in the sidebar
   - Select any PDF file from your computer
   - Wait for processing (you'll see a progress indicator)-> process is clearly visible in the terminal also
   - Once complete, the PDF is cached

2. **Ask Questions**
   - Type your question in the main area (e.g., "What is the main topic of this PDF?")
   - Press Enter or click the Submit button
   - The app will:
     - Show relevant page numbers instantly
     - Then display the AI-generated answer

3. **Manage PDFs**
   - In the sidebar, you'll see all previously processed PDFs
   - Click the (trash) icon next to any PDF to delete it

## Configuration

Edit config.py to customize:

```python
# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default LLM provider and model
DEFAULT_LLM_PROVIDER = "groq"
DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Retrieval: how many chunks to retrieve per query
RETRIEVER_TOP_K = 8

# Chunking: size and overlap of text chunks
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# OCR: minimum characters before falling back to OCR
MIN_PAGE_CHARS = 50
```

## Project Structure

### Key Files

| File               | Purpose                             |
| ------------------ | ----------------------------------- |
| app.py             | Streamlit UI and main event loop    |
| config.py          | Centralized settings                |
| ingestion.py       | PDF text extraction (PyMuPDF + OCR) |
| embedding_model.py | Embedding model initialization      |
| vectore_store.py   | Chroma vector store management      |
| llm.py             | LLM provider setup                  |
| retriever.py       | Two-step retrieval system           |
| prompt_design.py   | LLM prompt templates                |

### Directories

| Directory       | Purpose                                   |
| --------------- | ----------------------------------------- |
| embedding_cache | Downloaded embedding model weights        |
| vectorstores    | PDF vector databases (one folder per PDF) |
| data            | Sample data or inputs                     |
| notebooks       | Jupyter notebooks for experimentation     |

## Troubleshooting

### Issue: "ModuleNotFoundError" when running the app

**Solution:** Make sure your virtual environment is activated:

```bash
source chatwithpdfvenv/bin/activate  # macOS/Linux
chatwithpdfvenv/Scripts/activate #windows

conda activate chatwithpdfvenv #conda
```

### Issue: "GROQ_API_KEY not found"

**Solution:**
1. Create a .env file in your project root
2. Add: `GROQ_API_KEY=your_actual_key`
3. Restart the Streamlit app

### Issue: OCR not working / Tesseract error

**Solution:** Install Tesseract OCR (see Prerequisites section above)

### Issue: PDF not being processed correctly

**Possible causes:**

- PDF is corrupted — try another PDF
- PDF is scanned without text — OCR will handle it (slower)
- Very large PDF — may take time to process

### Issue: Slow response times

**Tips:**

- Reduce `RETRIEVER_TOP_K` in config.py (fewer chunks = faster)
- Use a smaller embedding model
- Use a faster LLM provider (Groq is typically faster than Gemini)

## Development Tips

### Working with Notebooks

```bash
#visit notebooks folder
jupyter notebook notebooks/
```

### Modifying Prompts

Edit prompt_design.py to change how the AI responds to questions.

## License

This project is open source. See LICENSE file for details.

## Support & Contributing

- Found a bug? Check existing issues
- Want to improve? Feel free to submit a pull request
- Have questions? Open an issue with details

---

**Happy PDf chatting!**

---
