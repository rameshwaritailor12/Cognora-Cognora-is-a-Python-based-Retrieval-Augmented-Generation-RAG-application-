# Cognora — AI-Powered Document Intelligence



Cognora is a Python-based Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents, ask questions in natural language, and explore relevant source passages.

The application combines semantic search with a locally hosted Large Language Model (LLM) to generate answers using retrieved document content.

## Features

- **PDF upload:** Upload a text-based PDF through the Streamlit interface.
- **Text extraction:** Extract text page by page while preserving PDF page numbers.
- **Document chunking:** Divide extracted text into smaller passages for retrieval.
- **Semantic search:** Generate text embeddings using Sentence Transformers and retrieve relevant passages using vector similarity.
- **LLM integration:** Use Llama 3.2 3B through Ollama to generate answers from retrieved context.
- **Source references:** Display retrieved passages and their PDF page numbers for verification.
- **Interactive interface:** Use a Streamlit chat interface with conversation history and custom CSS styling.
- **Local inference:** Run the language model locally through Ollama rather than using a paid cloud LLM API.

## Tech Stack

| Technology            | Purpose                                |
| --------------------- | -------------------------------------- |
| Python                | Application logic and RAG pipeline     |
| Streamlit             | Interactive web interface              |
| PyPDF                 | PDF text extraction                    |
| Sentence Transformers | Text embedding generation              |
| all-MiniLM-L6-v2      | Embedding model for semantic retrieval |
| NumPy                 | Vector similarity calculations         |
| Ollama                | Local LLM runtime                      |
| Llama 3.2 3B          | Answer generation                      |

## Architecture

1. **Upload:** The user uploads a PDF.
2. **Extract:** Text is extracted from each page.
3. **Chunk:** The extracted text is divided into smaller passages.
4. **Embed:** Sentence Transformers converts each passage into a vector embedding.
5. **Retrieve:** The user's question is embedded and compared with document embeddings using cosine similarity.
6. **Generate:** The most relevant passages are passed to the local LLM as context.
7. **Display:** Cognora presents the answer and retrieved source passages with page references.

See [`docs/architecture.md`](docs/architecture.md) for the architecture diagram.

## Project Structure

```text
Cognora/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── docs/
│   └── architecture.md
└── screenshots/
    ├── cognora-home.png
    └── cognora-answer.png
```

## Prerequisites

Install the following before running the application:

- Python 3.11 or another Python version compatible with the installed dependencies.
- Git.
- Ollama for your operating system.
- The `llama3.2:3b` model downloaded through Ollama.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Cognora
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from [https://ollama.com/download/windows](https://ollama.com/download/windows).

Download the model:

```powershell
ollama pull llama3.2:3b
```

Verify the model:

```powershell
ollama run llama3.2:3b
```

Enter a test prompt, then type `/bye` to exit. Ensure the Ollama service is running before using Cognora.

### 5. Launch Cognora

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Launch the application.
2. Upload a text-based PDF document.
3. Wait for the document to be processed.
4. Ask a question about the document.
5. Review the generated answer.
6. Expand **View retrieved PDF passages** to inspect the evidence and page references.

## Evaluation and Reliability

Evaluate the application using questions whose answers are present in the PDF, questions requiring information from multiple passages, and questions whose answers are absent from the document.

Check whether retrieved passages are relevant, answers are supported by the source text, and page references are correct.

Document evaluation results only after running and recording the tests.

## Limitations

- Scanned PDFs may require an OCR component before their contents can be retrieved.
- Generated answers may be incomplete or incorrect and should be checked against the source passages.
- Page references identify retrieved passages; they do not guarantee that every claim in an answer is supported.
- The local LLM and embedding model require disk space and system resources.
- The application is a portfolio prototype and has not been presented as a production-grade, multi-user service.






*Built as a hands-on project to explore Generative AI, semantic retrieval, and Retrieval-Augmented Generation.*
