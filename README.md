# ClauseInsight - LegalTech RAG System

A sophisticated Legal Document Analysis system using Retrieval-Augmented Generation (RAG) to help users locate and understand specific clauses within legal contracts.

## 🎯 Features

- **PDF Document Upload & Processing**: Upload legal contracts and automatically extract clauses
- **Semantic Search**: Find relevant clauses using natural language queries
- **AI-Powered Explanations**: Get plain-English explanations of complex legal clauses
- **Risk Analysis**: Understand potential risks and implications
- **Party Analysis**: See which party a clause favors
- **Beautiful Brutalist UI**: Modern, accessible interface with new brutalism design

## 🏗️ Architecture

### Backend (FastAPI + RAG Pipeline)
- **Document Processing**: Loads PDFs and splits into semantic chunks
- **Embeddings**: Uses Sentence Transformers for semantic embeddings
- **Vector Store**: FAISS for efficient similarity search
- **LLM Integration**: OpenAI GPT for structured clause explanations

### Frontend (React + TypeScript + Vite)
- **Modern Stack**: React 18, TypeScript, Tailwind CSS
- **UI Components**: Shadcn/ui with brutalist design
- **State Management**: React hooks for simple, effective state
- **API Integration**: Clean service layer for backend communication

## 🚀 Quick Start

### Backend Setup

1. Navigate to backend directory:
```bash
cd clause-insight-be
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

3. Create `.env` file from example:
```bash
copy .env.example .env  # Windows
# or
cp .env.example .env    # Linux/Mac
```

4. Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

5. Start the backend server:
```bash
cd app
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or from the root directory:
```bash
python -m app.main
```

Backend will be available at: http://localhost:8000

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd clause-insight-fe
```

2. Install dependencies:
```bash
npm install
# or
bun install
```

3. Start development server:
```bash
npm run dev
# or
bun dev
```

Frontend will be available at: http://localhost:5173

## 📖 Usage

1. **Start Both Servers**: Ensure both backend (port 8000) and frontend (port 5173) are running

2. **Upload a Contract**: 
   - Click or drag-and-drop a PDF contract into the upload area
   - Wait for processing to complete (you'll see a success message)

3. **Query Clauses**:
   - Enter a natural language question (e.g., "What are the termination conditions?")
   - Press Enter or click Search
   - View the retrieved clause with AI explanation

4. **Analyze Results**:
   - Review the relevance score and matched terms
   - Read the plain-English explanation
   - Understand potential risks
   - See which party the clause favors

## 🛠️ API Endpoints

### POST `/upload`
Upload and process a PDF document
- **Body**: Multipart form data with `file` field
- **Response**: Processing status and chunk count

### POST `/query`
Query indexed documents
- **Body**: 
  ```json
  {
    "query": "your question here",
    "top_k": 3
  }
  ```
- **Response**: 
  ```json
  {
    "clause": {
      "title": "Clause Title",
      "section": "Document Section",
      "content": "Full clause text"
    },
    "explanation": {
      "summary": "Brief summary",
      "meaning": "Plain English explanation",
      "risks": ["Risk 1", "Risk 2"],
      "favoredParty": "Neutral",
      "keyTerms": ["Term 1", "Term 2"]
    },
    "relevance": {
      "score": 95,
      "matchedTerms": ["term1", "term2"]
    }
  }
  ```

### GET `/health`
Check system health and index status

## 🔧 Configuration

### Backend (.env)
```
OPENAI_API_KEY=your_key
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
PORT=8000
```

### Frontend (.env)
```
VITE_API_URL=http://localhost:8000
```

## 📁 Project Structure

```
clause-insight-be/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── query.py         # Query endpoint
│   │   └── document.py      # Document upload endpoint
│   ├── rag/
│   │   ├── loader.py        # PDF loading
│   │   ├── splitter.py      # Text chunking
│   │   ├── embedder.py      # Embedding generation
│   │   ├── vector_store.py  # FAISS operations
│   │   └── generator.py     # LLM response generation
│   └── data/
│       ├── contracts/       # Uploaded PDFs
│       └── faiss_index/     # Vector index
└── requirements.txt

clause-insight-fe/
├── src/
│   ├── components/          # React components
│   ├── pages/              # Page components
│   ├── lib/
│   │   ├── api.ts          # API service layer
│   │   └── utils.ts        # Utilities
│   └── hooks/              # Custom React hooks
└── package.json
```

## 🎨 Design Philosophy

The frontend uses **New Brutalism** design principles:
- Bold borders and high contrast
- Flat colors with strategic use of shadows
- Uppercase tracking for headers
- Clear visual hierarchy
- No rounded corners (mostly)
- Functional, honest design

## 🔐 Security Notes

- Never commit `.env` files
- Keep your OpenAI API key secure
- Validate all file uploads
- Use HTTPS in production
- Implement rate limiting for production use

## 🤝 Contributing

This is a demonstration project. Feel free to fork and customize for your needs.

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **LangChain** for RAG framework
- **FAISS** for vector similarity search
- **Sentence Transformers** for embeddings
- **FastAPI** for backend framework
- **Shadcn/ui** for UI components
- **OpenAI** for LLM capabilities
