# HR CV Scanner — Project Reference

> Tài liệu này tóm tắt toàn bộ kiến trúc, codebase, và cách vận hành project để AI assistant có thể làm việc hiệu quả mà không cần đọc lại từng file.

---

## Mục đích

**HR CV Scanner** là hệ thống sàng lọc CV tự động dùng **multi-agent AI**. HR upload batch CVs (PDF/DOCX), hệ thống tự động:
1. Parse CV → structured data
2. Match CV với Job Description (RAG)
3. Chấm điểm theo rubric (0–100)
4. Xuất báo cáo PDF có bảng xếp hạng ứng viên

---

## Tech Stack

| Layer | Technology | Ghi chú |
|---|---|---|
| LLM | Google Gemini (`gemini-2.5-flash`) | Free tier: 15 req/min, 1500 req/day |
| Agent Orchestration | LangGraph | State machine 4 nodes |
| Vector DB | ChromaDB (local persistent) | `data/chroma/` |
| Embedding | `all-MiniLM-L6-v2` (local) | ~80MB, 384 dim, không tốn API |
| CV Parsing | pdfplumber, python-docx, pytesseract (OCR fallback) | |
| PDF Report | ReportLab | Deterministic, không dùng LLM |
| Backend | FastAPI + uvicorn | Port 8000 |
| Frontend | Chainlit (chat UI) | Port 8001 |
| Cloud (optional) | AWS S3 + boto3 | Fallback lưu local nếu không có S3 |
| Validation | Pydantic v2 | Mọi agent input/output đều typed |

---

## Cấu trúc thư mục

```
hr-cv-scanner/
├── agents/                  # 4 AI agents chính
│   ├── cv_parser.py         # Agent 1: Parse CV → CandidateProfile
│   ├── jd_matcher.py        # Agent 2: Match CV vs JD (RAG + LLM)
│   ├── scorer.py            # Agent 3: Chấm điểm (hybrid: toán + LLM)
│   └── report_writer.py     # Agent 4: Tạo PDF report (không dùng LLM)
│
├── graph/                   # LangGraph pipeline
│   ├── state.py             # GraphState schema (Pydantic)
│   ├── nodes.py             # 4 node functions + conditional edges
│   └── workflow.py          # build_graph(), run_pipeline(), stream_pipeline()
│
├── rag/                     # RAG layer
│   ├── embedder.py          # sentence-transformers wrapper (lru_cache)
│   ├── ingestor.py          # Ingest JD + rubric → ChromaDB
│   └── retriever.py         # search_jd_requirements(), get_rubric()
│
├── api/                     # FastAPI backend
│   ├── main.py              # App entry, lifespan, CORS, shared state
│   └── routes/
│       ├── jobs.py          # POST/GET/DELETE /api/v1/jobs
│       ├── scan.py          # POST /start, GET /status, GET /stream (SSE)
│       └── reports.py       # GET /run/{run_id}/download|shortlist
│
├── schemas/                 # Pydantic data contracts
│   ├── cv_schema.py         # CandidateProfile, WorkEntry, EducationEntry...
│   ├── match_schema.py      # MatchResult, SkillGap, ExperienceMatch...
│   ├── score_schema.py      # CandidateScore, RankedCandidate, ScoringRubric...
│   └── report_schema.py     # ReportOutput, ReportMeta
│
├── prompts/                 # LLM system prompts
│   ├── parser_prompt.py     # PARSER_SYSTEM_PROMPT, PARSER_USER_TEMPLATE
│   ├── matcher_prompt.py    # MATCHER_SYSTEM_PROMPT, MATCHER_USER_TEMPLATE
│   ├── scorer_prompt.py     # SCORER_SYSTEM_PROMPT, SCORER_USER_TEMPLATE
│   └── writer_prompt.py     # (không dùng LLM, giữ cho consistent)
│
├── tools/                   # Utility tools
│   ├── pdf_extractor.py     # pdfplumber + pytesseract OCR fallback
│   ├── docx_extractor.py    # python-docx extractor
│   ├── pdf_generator.py     # ReportLab PDF generator
│   └── s3_uploader.py       # boto3 S3 upload + save_local fallback
│
├── frontend/
│   └── app.py               # Chainlit chat UI (port 8001)
│
├── data/
│   ├── chroma/              # ChromaDB persistent storage
│   └── CV_*.pdf/.docx       # 20 sample CVs tiếng Việt
│
├── tests/                   # pytest test suite
├── create_jd.py             # Script tạo JD mẫu (Senior Backend Engineer)
├── requirements.txt         # Dependencies
├── pyproject.toml           # Project metadata + dev deps
├── Dockerfile               # Docker image (chỉ backend port 8000)
├── .env.example             # Template env vars
└── chainlit.md              # Chainlit welcome screen
```

---

## Pipeline Architecture

```
START
  │
  ▼
parse_node (Agent 1: CVParserAgent)
  │  Input:  file_paths, api_key
  │  Output: parsed_candidates: list[CandidateProfile]
  │          progress: 0.25
  │
  ├─ [FAILED] → END
  ▼
match_node (Agent 2: JDMatcherAgent)
  │  Input:  parsed_candidates, job_id, job_title, api_key
  │  Output: match_results: list[MatchResult]
  │          progress: 0.55
  │
  ├─ [FAILED] → END
  ▼
score_node (Agent 3: ScorerAgent)
  │  Input:  match_results, parsed_candidates, job_id, api_key
  │  Output: ranked_candidates: list[RankedCandidate]
  │          progress: 0.80
  │
  ├─ [FAILED] → END
  ▼
report_node (Agent 4: ReportWriterAgent)
  │  Input:  ranked_candidates, job_title, job_id
  │  Output: report: ReportOutput
  │          progress: 1.0
  ▼
END
```

### GraphState (graph/state.py)

```python
class GraphState(BaseModel):
    # INPUT
    file_paths:  list[str]
    job_id:      str
    job_title:   str
    api_key:     str           # Gemini API key

    # PROGRESS
    status:      str           # PipelineStatus enum
    current_step: str          # Log cho Chainlit stream
    progress:    float         # 0.0 → 1.0

    # NODE OUTPUTS (Annotated với operator.add → list append, không replace)
    parsed_candidates: Annotated[list[CandidateProfile], operator.add]
    parse_errors:      Annotated[list[str], operator.add]
    match_results:     Annotated[list[MatchResult], operator.add]
    ranked_candidates: list[RankedCandidate]
    report:            Optional[ReportOutput]
    errors:            Annotated[list[str], operator.add]
```

---

## Data Contracts (schemas/)

### CandidateProfile (cv_schema.py)
Output của Agent 1, input của Agent 2 + 3.

| Field | Type | Ghi chú |
|---|---|---|
| `full_name` | str | |
| `contact` | ContactInfo | email, phone, linkedin, github, location |
| `total_experience_years` | float\|None | Chỉ tính professional roles |
| `work_history` | list[WorkEntry] | company, role, duration_months, achievements, technologies |
| `technical_skills` | list[str] | |
| `soft_skills` | list[str] | |
| `certifications` | list[str] | |
| `education` | list[EducationEntry] | institution, degree, level, gpa |
| `confidence` | ParseConfidence | HIGH/MEDIUM/LOW |
| `extraction_method` | str | native_pdf / ocr / docx |

### MatchResult (match_schema.py)
Output của Agent 2, input của Agent 3.

Key fields: `skill_gap` (matched, missing_critical, bonus), `experience` (candidate_years, required_years, domain_relevance), `match_summary`, `low_confidence`, `warnings`.

### CandidateScore & RankedCandidate (score_schema.py)
Output của Agent 3, input của Agent 4.

- `total_score`: 0–100
- `tier`: strong (≥80) / good (60-79) / moderate (40-59) / weak (<40)
- `breakdown`: DimensionScore cho 5 dimensions
- `rank`, `percentile` (trong RankedCandidate)

### Default Rubric (5 dimensions, tổng = 100)
| Dimension | Weight | Scored by |
|---|---|---|
| technical_skills | 35 | Programmatic (đếm khớp skill) |
| experience | 20 | Programmatic (year ratio × domain relevance) |
| education | 15 | LLM |
| achievements | 20 | LLM |
| soft_skills | 10 | LLM |

---

## Agents Chi Tiết

### Agent 1 — CVParserAgent (agents/cv_parser.py)

- Model: `gemini-2.5-flash`, temperature=0, max_tokens=4096
- MAX_RETRIES = 3, RETRY_DELAY = 2s
- Free tier: sleep 4s giữa các calls trong batch
- Confidence heuristic: -20 nếu OCR, -20 nếu no name, -20 nếu no work_history, -15 nếu no skills
- `_fallback_profile()`: không crash batch khi parse lỗi

**Flow:** file → `_extract_text()` → `_call_llm_with_retry()` → `_parse_json_response()` → `CandidateProfile.model_validate()`

### Agent 2 — JDMatcherAgent (agents/jd_matcher.py)

- Model: `gemini-3.1-flash-lite-preview`, temperature=0
- Retrieve top_k=12 chunks từ ChromaDB theo cosine similarity
- Query text: tóm tắt CV (skills + 3 recent roles + education)
- Prompt format: CV JSON + JD requirements có [REQUIRED]/[NICE] tag + relevance score
- `low_confidence` inherited từ CVParserAgent nếu parse confidence = LOW

### Agent 3 — ScorerAgent (agents/scorer.py)

- **Hybrid scoring** — không để LLM chấm hết (LLM drift problem):
  - `technical_skills`: đếm matched/total × weight, penalty -12% mỗi critical skill bị thiếu
  - `experience`: year_ratio × 0.6 + domain_relevance × 0.4 (max 1.2× bonus cho over-qualified)
  - `education`, `achievements`, `soft_skills`: 1 LLM call cho tất cả 3 dimensions
- Rubric load từ ChromaDB, fallback về DEFAULT_RUBRIC
- Ranking: sort giảm dần total_score, tính percentile = (total - rank + 1) / total × 100

### Agent 4 — ReportWriterAgent (agents/report_writer.py)

- **Không dùng LLM** — deterministic hoàn toàn
- Tạo PDF với ReportLab (`tools/pdf_generator.py`)
- Upload S3 nếu có `S3_BUCKET_NAME` env, ngược lại lưu `reports/` local
- Shortlist: top 5 candidates có tier STRONG hoặc GOOD
- Summary text: template-based

---

## RAG Layer

### Embedder (rag/embedder.py)
- Model: `all-MiniLM-L6-v2` (local, không tốn API)
- `lru_cache(maxsize=1)` — load 1 lần duy nhất
- `embed_text(text)` → list[384 floats]
- `embed_batch(texts)` → list[list[float]], batch_size=32

### Ingestor (rag/ingestor.py)
ChromaDB collections:
- `job_descriptions`: mỗi JD requirement = 1 chunk, metadata: `{job_id, job_title, type, priority, index}`
- `scoring_rubrics`: 1 document nguyên JSON per job_id

Key functions:
- `ingest_job_description(job_id, job_title, requirements, nice_to_have)` → int (chunk count)
- `ingest_rubric(job_id, rubric_dict)` → None
- `list_jobs()` → list[{job_id, job_title}]
- `clear_job(job_id)` → None

Chunk ID format: `{job_id}_req_{i:03d}` / `{job_id}_nice_{i:03d}`

### Retriever (rag/retriever.py)
- `search_jd_requirements(query_text, job_id, top_k=10, min_score=0.3)` → list[RetrievedChunk]
- Score = `1.0 - distance / 2.0` (chromadb trả về cosine distance)
- Filter theo `job_id` để chỉ search trong 1 JD
- `get_rubric(job_id)` → dict | None

---

## API Endpoints (FastAPI port 8000)

### Jobs — `/api/v1/jobs`
| Method | Path | Mô tả |
|---|---|---|
| POST | `/` | Ingest JD mới vào ChromaDB |
| GET | `/` | List tất cả JDs |
| DELETE | `/{job_id}` | Xóa JD |
| POST | `/{job_id}/rubric` | Upload scoring rubric (weights phải = 100) |

### Scan — `/api/v1/scan`
| Method | Path | Mô tả |
|---|---|---|
| POST | `/start` | Upload CVs (multipart), start pipeline background → trả run_id |
| GET | `/status/{run_id}` | Poll status (status, progress, step) |
| GET | `/stream/{run_id}` | SSE stream real-time progress (Chainlit dùng) |

Form fields cho `/start`: `files` (UploadFile[]), `job_id`, `job_title`, `api_key`
Constraints: `.pdf/.docx/.doc` only, max 10MB/file

### Reports — `/api/v1/reports`
| Method | Path | Mô tả |
|---|---|---|
| GET | `/run/{run_id}` | Full report (summary + shortlist + metadata) |
| GET | `/run/{run_id}/download` | Download PDF binary |
| GET | `/run/{run_id}/shortlist` | Top 5 candidates nhanh (Chainlit dùng) |

### Health
- `GET /health` → `{status, pipeline_ready, active_scans}`

### In-memory Store (api/main.py)
```python
scan_jobs: dict[str, Any] = {
    run_id: {
        "status":    "pending|running|completed|failed",
        "progress":  float,
        "step":      str,
        "result":    ReportOutput | None,
        "error":     str | None,
        "job_id":    str,
        "job_title": str,
        "file_count": int,
    }
}
```
**Lưu ý:** In-memory, mất khi restart server. Production cần DynamoDB/Redis.

---

## Frontend — Chainlit (frontend/app.py, port 8001)

Chat UI với các commands:
| Command | Mô tả |
|---|---|
| `/setjob <id> <title>` | Chọn job để scan |
| `/setkey <api_key>` | Set Gemini API key |
| `/jobs` | Xem danh sách jobs |
| `/scan` | Bắt đầu scan (cần đã upload CV và setjob) |
| `/status <run_id>` | Kiểm tra tiến trình |

Upload CV: dùng nút 📎 (Chainlit file element). Sau khi upload có Action buttons "🚀 Bắt đầu Scan" / "❌ Huỷ".

API_BASE: `http://localhost:8000/api/v1` (env: `API_BASE_URL`)

SSE flow: Chainlit → `/scan/stream/{run_id}` → parse `data: {json}` → update message real-time → khi `status=completed` → gọi `/reports/run/{run_id}/shortlist` → hiện kết quả + Action "Download PDF".

---

## Scoring Formula

### Technical Skills
```
match_ratio = len(matched) / (len(matched) + len(missing_critical))
effective_ratio = max(0, match_ratio - len(missing_critical) × 0.12)
raw_score = effective_ratio × 35
```

### Experience
```
year_ratio  = min(candidate_years / required_years, 1.2)
year_score  = year_ratio × 0.6
domain_score = domain_relevance × 0.4   # domain_relevance: 0.0–1.0 từ LLM
effective_ratio = min(year_score + domain_score, 1.0)
raw_score = effective_ratio × 20
```

### Tier Classification
| Tier | Điều kiện | Ý nghĩa |
|---|---|---|
| 🟢 STRONG | ≥ 80 | Ưu tiên phỏng vấn |
| 🔵 GOOD | 60–79 | Cân nhắc phỏng vấn |
| 🟡 MODERATE | 40–59 | Xem xét kỹ hơn |
| 🔴 WEAK | < 40 | Không phù hợp |

---

## Environment Variables

```env
# Bắt buộc
GEMINI_API_KEY=your_gemini_api_key

# Optional — có giá trị mặc định
CHROMA_DB_PATH=data/chroma
API_BASE_URL=http://localhost:8000/api/v1   # Chainlit → FastAPI

# Optional — AWS (có fallback local nếu không set)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_REPORT_BUCKET=hr-cv-scanner-reports
S3_BUCKET_NAME=...   # Nếu set → ReportWriterAgent upload S3
```

---

## Cách chạy

### Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Sửa .env: thêm GEMINI_API_KEY
```

### Tạo JD mẫu (phải chạy trước khi scan)
```powershell
# Cần FastAPI đang chạy trước
python create_jd.py
# Tạo JD: job_id="backend-2025", job_title="Senior Backend Engineer"
```

Hoặc dùng API trực tiếp:
```bash
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{"job_id":"backend-2025","job_title":"Senior Backend Engineer","requirements":["3+ years Python","FastAPI experience","AWS Lambda"]}'
```

### Chạy Backend
```powershell
uvicorn api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```

### Chạy Frontend
```powershell
# Terminal mới
chainlit run frontend/app.py --port 8001
# UI: http://localhost:8001
```

### Docker (chỉ backend)
```powershell
docker build -t hr-cv-scanner .
docker run -p 8000:8000 --env-file .env hr-cv-scanner
```

---

## Data Mẫu

`data/` có sẵn **20 CV tiếng Việt** để test:
- CV_01 → CV_10: `.docx` format
- CV_11 → CV_20: `.pdf` format

---

## Luồng hoạt động đầy đủ

```
1. [Setup] ingest_job_description() → ChromaDB
           ingest_rubric()          → ChromaDB (optional, dùng DEFAULT nếu không có)

2. [Request] POST /api/v1/scan/start
             - Validate files (pdf/docx, ≤10MB)
             - Lưu vào tmpdir
             - Tạo run_id (8 chars UUID uppercase)
             - Background task: _run_pipeline_background()
             - Return 202 + {run_id, stream_url}

3. [Pipeline] stream_pipeline(graph, file_paths, job_id, job_title, api_key)
             → parse_node:  CVParserAgent.parse() mỗi file → CandidateProfile[]
             → match_node:  JDMatcherAgent.match_batch() → MatchResult[]
             → score_node:  ScorerAgent.score_and_rank() → RankedCandidate[]
             → report_node: ReportWriterAgent.write() → ReportOutput (PDF)

4. [Monitor] GET /api/v1/scan/stream/{run_id}  ← SSE, Chainlit subscribe
             GET /api/v1/scan/status/{run_id}  ← Poll alternative

5. [Result]  GET /api/v1/reports/run/{run_id}/shortlist  ← Quick summary
             GET /api/v1/reports/run/{run_id}/download   ← PDF bytes
```

---

## Patterns & Conventions

### LLM Retry Pattern (tất cả 3 agents đều dùng)
```python
MAX_RETRIES = 3
for attempt in range(1, MAX_RETRIES + 1):
    try:
        raw = self._call_llm(prompt)
        return self._parse_json_response(raw)
    except (json.JSONDecodeError, ValueError) as e:
        if attempt < MAX_RETRIES:
            prompt += "\n\nIMPORTANT: Output ONLY raw JSON..."
            time.sleep(RETRY_DELAY)
        else:
            raise RuntimeError(...) from e
```

### JSON Response Parsing (nhất quán)
```python
text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
text = re.sub(r"\s*```$",           "", text, flags=re.MULTILINE)
if not text.startswith("{"):
    text = text[text.find("{"):]
return json.loads(text)
```

### Rate Limiting (Gemini free tier 15 req/min)
```python
if i < len(items):
    time.sleep(4)  # ~4s giữa các calls trong batch
```

### Fallback Pattern (không crash batch)
Mỗi agent có `_fallback_*()` method trả về minimal object khi xử lý lỗi.

---

## Tests

```
tests/
├── test_api.py           # FastAPI endpoint tests (httpx TestClient)
├── test_cv_parser.py     # CVParserAgent unit tests
├── test_jd_matcher.py    # JDMatcherAgent unit tests
├── test_scorer.py        # ScorerAgent unit tests
├── test_report_writer.py # ReportWriterAgent unit tests
├── test_retriever.py     # RAG retriever tests
└── test_orchestrator.py  # End-to-end pipeline tests
```

```powershell
pytest tests/ -v
```

---

## Quan trọng khi sửa code

1. **Thêm JD mới**: Gọi `ingest_job_description()` + `ingest_rubric()`. Không chỉnh ChromaDB trực tiếp.
2. **Thay LLM**: Chỉ cần thay `model` param trong constructor của mỗi agent. Interface `_call_llm()` giữ nguyên.
3. **Thêm scoring dimension**: Sửa `ScoringRubric`, `ScoreBreakdown`, `DEFAULT_RUBRIC`, và logic trong `ScorerAgent`.
4. **Scale up**: LangGraph `Send()` API để fan-out parse/match theo từng file parallel thay vì sequential.
5. **Production storage**: Thay `scan_jobs` dict bằng DynamoDB/Redis. PDF lưu S3 (set `S3_BUCKET_NAME`).
6. **CORS**: Backend allow origins `localhost:8001` (Chainlit). Thêm production domain vào `allow_origins`.
7. **Windows async**: `frontend/app.py` dòng đầu set `WindowsSelectorEventLoopPolicy` cho Python 3.11+ trên Windows.
