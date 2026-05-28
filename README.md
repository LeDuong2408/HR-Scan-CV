# HR CV Scanner

Scaffold for a multi-agent CV screening platform using FastAPI, LangGraph, RAG (ChromaDB), and Chainlit.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run API:

```bash
uvicorn api.main:app --reload --port 8000
```

Run UI:

```bash
chainlit run frontend/app.py -w
```
```bash
streamlit run frontend/streamlit_app.py
```

Run tests:

```bash
pytest
```

## Workflow Overview

1. Parse CV (`agents/cv_parser.py`)
2. Retrieve JD/rubric context (`agents/jd_matcher.py`)
3. Score with rubric (`agents/scorer.py`)
4. Write PDF report (`agents/report_writer.py`)

Orchestrated in `graph/workflow.py`.

