# HR CV Scanner — Project Reference

> Tài liệu tham chiếu toàn bộ kiến trúc, codebase, và cách vận hành.
> Cập nhật lần cuối: phản ánh Agent 1 v2 (Markdown+Chunking) và Agent 2 v2 (JD parse + RAG evidence).

---

## Mục đích

**HR CV Scanner** là hệ thống sàng lọc CV tự động dùng **multi-agent AI**. HR upload batch CVs (PDF/DOCX) + paste JD text, hệ thống tự động:
1. Parse CV → Markdown → Chunks → ChromaDB (không dùng LLM)
2. Parse JD → query ChromaDB evidence per skill → Match analysis
3. Chấm điểm hybrid (programmatic + LLM) theo rubric (0–100)
4. Xuất báo cáo PDF có bảng xếp hạng ứng viên

---

## Tech Stack

| Layer | Technology | Ghi chú |
|---|---|---|
| LLM | Google Gemini (`gemini-1.5-flash`) | Free tier: 15 req/min, 1500 req/day |
| Agent Orchestration | LangGraph | State machine 4 nodes |
| Vector DB | ChromaDB (local persistent) | `data/chroma/` — 2 collections |
| Embedding | `all-MiniLM-L6-v2` (local) | ~80MB, 384 dim, không tốn API |
| CV → Markdown | pymupdf4llm (PDF), mammoth (DOCX) | Giữ heading structure |
| Chunking | LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter | Max 900 chars/chunk |
| PDF Report | ReportLab | Deterministic, không dùng LLM |
| Backend | FastAPI + uvicorn | Port 8000 |
| Frontend | Streamlit | Port 8501 (Chainlit có bug trên Windows/Python 3.14) |
| Tracing | LangSmith (optional) | Free tier: 5,000 traces/tháng |
| Cloud (optional) | AWS S3 + boto3 | Fallback lưu local nếu không có S3 |
| Validation | Pydantic v2 | Mọi agent input/output đều typed |

---

## Cấu trúc thư mục

```
hr-cv-scanner/
├── agents/
│   ├── cv_parser.py         # Agent 1 v2: CV → Markdown → Chunks → ChromaDB → ParsedCV
│   ├── jd_matcher.py        # Agent 2 v2: JD parse + RAG evidence + LLM analyze → MatchResult
│   ├── scorer.py            # Agent 3: Chấm điểm hybrid (programmatic + LLM)
│   └── report_writer.py     # Agent 4: Tạo PDF report (không dùng LLM)
│
├── graph/
│   ├── state.py             # GraphState schema — có thêm jd_text field
│   ├── nodes.py             # 4 node functions + conditional edges (cập nhật v2)
│   └── workflow.py          # build_graph(), run_pipeline(), stream_pipeline()
│
├── rag/
│   ├── embedder.py          # sentence-transformers wrapper (lru_cache)
│   ├── ingestor.py          # Ingest JD + rubric → ChromaDB (collection: job_descriptions)
│   ├── retriever.py         # search_jd_requirements(), get_rubric()
│   └── cv_chunker.py        # NEW: chunk_and_store(), query_cv_chunks() (collection: cv_chunks)
│
├── api/
│   ├── main.py              # App entry, lifespan, CORS, LangSmith setup
│   ├── tracer.py            # NEW: LangSmith setup helpers
│   └── routes/
│       ├── jobs.py          # POST/GET/DELETE /api/v1/jobs + rubric
│       ├── scan.py          # POST /start (có jd_text field), GET /status, GET /stream (SSE)
│       └── reports.py       # GET /run/{id}/download|shortlist
│
├── schemas/
│   ├── cv_schema.py         # CandidateProfile (legacy, dùng cho test cũ)
│   ├── jd_schema.py         # NEW: ParsedJD (output của LLM parse JD)
│   ├── match_schema.py      # MatchResult, SkillGap, ExperienceMatch
│   ├── score_schema.py      # CandidateScore, RankedCandidate, ScoringRubric
│   └── report_schema.py     # ReportOutput, ReportMeta
│
├── prompts/
│   ├── parser_prompt.py     # (legacy — Agent 1 v2 không dùng LLM)
│   ├── matcher_prompt.py    # JD_PARSE_SYSTEM_PROMPT + MATCH_ANALYZE_SYSTEM_PROMPT (2 prompts)
│   └── scorer_prompt.py     # SCORER_SYSTEM_PROMPT, SCORER_USER_TEMPLATE
│
├── tools/
│   ├── cv_to_markdown.py    # NEW: PDF/DOCX → Markdown (pymupdf4llm + mammoth)
│   ├── pdf_extractor.py     # pdfplumber + pytesseract OCR fallback (dùng bởi cv_to_markdown)
│   ├── docx_extractor.py    # python-docx extractor (legacy)
│   ├── pdf_generator.py     # ReportLab PDF generator
│   └── s3_uploader.py       # boto3 S3 upload + save_local fallback
│
├── frontend/
│   ├── streamlit_app.py     # Streamlit UI — DÙNG CÁI NÀY (ổn định trên Windows)
│   └── app.py               # Chainlit UI — có bug trên Windows/Python 3.14, bỏ qua
│
├── data/
│   ├── chroma/              # ChromaDB persistent storage (2 collections)
│   │   ├── job_descriptions # JD requirements + rubrics
│   │   └── cv_chunks        # CV chunks với cv_id metadata
│   └── CV_*.pdf/.docx       # Sample CVs để test
│
├── tests/
│   ├── test_cv_parser_v2.py # Agent 1 v2 tests (markdown, chunking, heuristics)
│   ├── test_jd_matcher_v2.py# Agent 2 v2 tests (JD parse, evidence query, match)
│   ├── test_scorer.py       # Agent 3 tests
│   ├── test_report_writer.py# Agent 4 tests
│   ├── test_orchestrator.py # LangGraph pipeline tests
│   └── test_api.py          # FastAPI endpoint tests
│
├── PROJECT.md               # File này
├── requirements.txt
└── .env.example
```

---

## Pipeline Architecture (v2)

```
START
  │
  ▼
parse_node (Agent 1 v2: CVParserAgent)
  │  Input:  file_paths
  │  Action: PDF/DOCX → Markdown (pymupdf4llm/mammoth)
  │           → Chunk theo H1 heading (max 900 chars)
  │           → Embed + store ChromaDB (collection: cv_chunks)
  │  Output: parsed_candidates: list[ParsedCV]  ← có cv_id để query ChromaDB
  │          progress: 0.25
  │  LLM calls: 0 (không tốn API quota)
  │
  ├─ [FAILED] → END
  ▼
match_node (Agent 2 v2: JDMatcherAgent)
  │  Input:  parsed_candidates, jd_text, job_id, job_title, api_key
  │  Action: Step 1 — LLM parse jd_text → ParsedJD (cache per job_id)
  │           Step 2 — Với MỖI required_skill: query cv_chunks của candidate
  │           Step 3 — LLM analyze evidence → MatchResult
  │  Output: match_results: list[MatchResult]
  │          progress: 0.55
  │  LLM calls: 1 (parse JD, cached) + 1 per candidate (analyze)
  │
  ├─ [FAILED] → END
  ▼
score_node (Agent 3: ScorerAgent)
  │  Input:  match_results, parsed_candidates, job_id, api_key
  │  Action: Programmatic score (technical + experience)
  │           + 1 LLM call per candidate (education + achievements + soft_skills)
  │           → Rank toàn batch
  │  Output: ranked_candidates: list[RankedCandidate]
  │          progress: 0.80
  │
  ├─ [FAILED] → END
  ▼
report_node (Agent 4: ReportWriterAgent)
  │  Input:  ranked_candidates, job_title, job_id
  │  Action: Generate PDF (ReportLab) → S3 hoặc local
  │  Output: report: ReportOutput
  │          progress: 1.0
  │  LLM calls: 0
  ▼
END
```

---

## GraphState (graph/state.py)

```python
class GraphState(BaseModel):
    # INPUT
    file_paths:  list[str]
    job_id:      str
    job_title:   str
    jd_text:     str    # NEW v2: Full JD raw text — Agent 2 parse thành ParsedJD
    api_key:     str    # Gemini API key

    # PROGRESS
    status:       str   # PipelineStatus: pending/parsing/matching/scoring/reporting/completed/failed
    current_step: str   # Log message cho Streamlit/Chainlit stream
    progress:     float # 0.0 → 1.0

    # NODE OUTPUTS (Annotated với operator.add → append, không replace)
    parsed_candidates: Annotated[list[ParsedCV], operator.add]  # v2: ParsedCV thay CandidateProfile
    parse_errors:      Annotated[list[str], operator.add]
    match_results:     Annotated[list[MatchResult], operator.add]
    ranked_candidates: list[RankedCandidate]
    report:            Optional[ReportOutput]
    errors:            Annotated[list[str], operator.add]
    retry_count:       int
```

---

## Data Contracts (schemas/)

### ParsedCV (agents/cv_parser.py) — Agent 1 v2 output
| Field | Type | Ghi chú |
|---|---|---|
| `cv_id` | str | `cv_{md5(filename)[:12]}` — idempotent, dùng để query ChromaDB |
| `file_name` | str | Tên file gốc |
| `candidate_name` | str | Extracted bằng heuristic từ markdown (không cần LLM) |
| `email` | str | Extracted bằng regex |
| `markdown` | str | Full markdown text của CV |
| `chunk_count` | int | Số chunks đã store vào ChromaDB |
| `sections` | list[str] | Sections detect được: ["Experience", "Skills", "Education"] |
| `parse_method` | str | "pymupdf" / "pymupdf_fallback" / "mammoth" / "failed" |
| `warnings` | list[str] | |

### ParsedJD (schemas/jd_schema.py) — Agent 2 internal
| Field | Type | Ghi chú |
|---|---|---|
| `job_title` | str | |
| `required_skills` | list[str] | Mỗi item ngắn gọn để query ChromaDB: ["Python 3+ years", "FastAPI"] |
| `nice_to_have` | list[str] | |
| `required_experience_years` | float\|None | |
| `required_experience_domain` | str\|None | "backend development" |
| `required_education_level` | str\|None | "bachelor" / "master" / "phd" / "any" |
| `seniority_level` | str\|None | "junior" / "mid" / "senior" / "lead" |

### MatchResult (schemas/match_schema.py) — Agent 2 output, Agent 3 input
Key fields: `requirement_matches` (per skill: evidence, match_level full/partial/missing), `skill_gap` (matched, missing_critical, missing_nice, bonus), `experience` (candidate_years, required_years, domain_relevance 0–1), `match_summary`, `low_confidence`, `warnings`.

### CandidateScore & RankedCandidate (schemas/score_schema.py) — Agent 3 output
- `total_score`: 0–100
- `tier`: strong (≥80) / good (60–79) / moderate (40–59) / weak (<40)
- `breakdown`: DimensionScore cho 5 dimensions
- `rank`, `percentile` (trong RankedCandidate)

---

## Agents Chi Tiết

### Agent 1 v2 — CVParserAgent (agents/cv_parser.py)

**Không gọi LLM** — nhanh hơn v1, không tốn API quota.

**Pipeline:**
```
file.pdf / file.docx
    ↓ pymupdf4llm / mammoth
Markdown (giữ # heading structure)
    ↓ MarkdownHeaderTextSplitter (langchain)
Chunks theo H1 section [Experience, Skills, Education...]
    ↓ RecursiveCharacterTextSplitter nếu chunk > 900 chars
Sub-chunks (≤ ~256 tokens cho all-MiniLM-L6-v2)
    ↓ embed_batch (local, free)
ChromaDB collection: cv_chunks
    metadata: {cv_id, file_name, section, subsection, chunk_index}
```

**Key design:**
- `cv_id = f"cv_{md5(filename)[:12]}"` — idempotent, cùng file → cùng cv_id → upsert không duplicate
- `MAX_CHUNK_CHARS = 900` — all-MiniLM-L6-v2 max 256 tokens ≈ 1024 chars, dùng 900 để có buffer
- `strip_headers=False` — giữ heading trong chunk text để LLM biết context
- `reprocess=True` trong parse_node — xóa chunks cũ trước khi process lại

### Agent 2 v2 — JDMatcherAgent (agents/jd_matcher.py)

**3 bước:**

**Bước 1 — Parse JD (LLM call #1, cache per job_id):**
```
jd_text → Gemini → ParsedJD
  {required_skills: ["Python 3+ years", "FastAPI", "AWS Lambda"...],
   nice_to_have: ["Kubernetes"],
   required_experience_years: 3.0, ...}
```
Cache trong `self._jd_cache[job_id]` — toàn batch 50 CVs chỉ gọi 1 lần.

**Bước 2 — Query ChromaDB evidence per skill (không LLM):**
```
# Query direction: JD skills → CV chunks (ĐÚNG)
for skill in parsed_jd.required_skills:
    chunks = query_cv_chunks(query_text=skill, cv_id=candidate.cv_id, top_k=3, min_score=0.0)
    evidence[skill] = chunks  # rỗng nếu không có → MISSING

# KHÔNG filter min_score — evidence rỗng = MISSING, LLM tự kết luận
```

**Bước 3 — Analyze (LLM call #2 per candidate):**
```
ParsedJD + cv_evidence_per_skill → Gemini → MatchResult
  {requirement_matches: [{requirement, candidate_evidence, match_level}...],
   skill_gap: {matched, missing_critical, missing_nice, bonus},
   experience: {candidate_years, required_years, domain_relevance},
   match_summary, low_confidence, warnings}
```

**Tại sao query direction mới đúng hơn:**
- Cũ (v1): CV skills → query JD requirements → miss skills candidate không có
- Mới (v2): JD skills → query CV chunks → luôn check TẤT CẢ requirements, evidence rỗng = MISSING

### Agent 3 — ScorerAgent (agents/scorer.py)

**Hybrid scoring** (nhận `list[ParsedCV]` thay `list[CandidateProfile]` từ v2):
- `technical_skills` (35pt): programmatic — đếm matched/total, penalty -12% mỗi critical gap
- `experience` (20pt): programmatic — year_ratio × 0.6 + domain_relevance × 0.4
- `education` + `achievements` + `soft_skills` (45pt): 1 LLM call cho cả 3

LLM context cho education/achievements: extract relevant sections từ `candidate.markdown` (không còn dùng structured fields như v1).

Rubric: load từ ChromaDB theo job_id, fallback về `DEFAULT_RUBRIC` nếu chưa ingest.

### Agent 4 — ReportWriterAgent (agents/report_writer.py)

**Không dùng LLM.** ReportLab tạo PDF gồm:
- Cover page: title, stats (total/strong/good/avg score)
- Executive summary: ranking table màu sắc (🟢🔵🟡🔴)
- Candidate sections: score bars, strengths/concerns, recommendation

Upload S3 nếu có `S3_BUCKET_NAME`, fallback lưu `reports/` local.

---

## RAG Layer

### ChromaDB Collections

**`job_descriptions`** — JD requirements (ingest 1 lần qua API):
- Chunk ID: `{job_id}_req_{i:03d}` / `{job_id}_nice_{i:03d}`
- Metadata: `{job_id, job_title, priority, index}`
- Dùng bởi: `search_jd_requirements()` trong retriever.py (Agent 2 v1 dùng, v2 không dùng trực tiếp)

**`cv_chunks`** — CV sections (tạo tự động khi Agent 1 parse):
- Chunk ID: `{cv_id}_chunk_{chunk_index:04d}`
- Metadata: `{cv_id, file_name, section, subsection, chunk_index, char_count}`
- Dùng bởi: `query_cv_chunks()` trong cv_chunker.py (Agent 2 v2 query per skill)

### Key RAG Functions

```python
# Agent 1 → ChromaDB
from rag.cv_chunker import chunk_and_store, query_cv_chunks, delete_cv_chunks

# Agent 2 query
query_cv_chunks(query_text="Python 3+ years", cv_id="cv_abc123", top_k=3, min_score=0.0)
# → [{"text": "4 years Python at FPT...", "section": "Experience", "score": 0.91}]

# Embedder (dùng chung)
from rag.embedder import embed_text, embed_batch  # local, free, lru_cache
```

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
| POST | `/start` | Upload CVs + jd_text, start pipeline → trả run_id |
| GET | `/status/{run_id}` | Poll status |
| GET | `/stream/{run_id}` | SSE stream real-time progress |

**Form fields cho `/start`:**
```
files[]   — UploadFile[] (.pdf/.docx/.doc, max 10MB/file)
job_id    — str
job_title — str
jd_text   — str  ← NEW v2: full JD text để Agent 2 parse
api_key   — str  (Gemini API key)
```

### Reports — `/api/v1/reports`
| Method | Path | Mô tả |
|---|---|---|
| GET | `/run/{run_id}` | Full report (summary + shortlist + metadata) |
| GET | `/run/{run_id}/download` | Download PDF binary |
| GET | `/run/{run_id}/shortlist` | Top 5 candidates nhanh |

### Health
`GET /health` → `{status, pipeline_ready, active_scans}`

---

## Frontend — Streamlit (frontend/streamlit_app.py)

**Dùng Streamlit** thay Chainlit vì Chainlit có bug trên Windows với Python 3.14 (`anyio.NoEventLoopError`).

Giao diện 3 tabs:
- **Scan CVs**: Upload CVs + paste JD text + nút bắt đầu + progress tracking
- **Kết quả**: Bảng xếp hạng + summary + LangSmith link
- **Download**: Download PDF report

Sidebar: Gemini API key, chọn job từ ChromaDB, backend status.

**Chạy:** `streamlit run frontend/streamlit_app.py`

**Session state keys:**
```python
"job_id", "job_title", "jd_text",  # ← jd_text là mới
"api_key", "run_id", "status"
```

---

## LangSmith Tracing (optional)

Setup trong `api/tracer.py`, kích hoạt khi có `LANGSMITH_API_KEY` trong `.env`.

Traces hiển thị:
```
Pipeline Run
├── node:parse
│   ├── CVParserAgent.parse (cv1.pdf) — no LLM call
│   └── CVParserAgent.parse (cv2.pdf)
├── node:match
│   ├── JDMatcherAgent.match (Nguyen Van A)
│   │   ├── LLM: JD_PARSE → ParsedJD  (1 call, cached)
│   │   └── LLM: MATCH_ANALYZE → MatchResult
│   └── JDMatcherAgent.match (Tran Thi B)
│       └── LLM: MATCH_ANALYZE → MatchResult  (JD parse cached)
└── node:score / node:report
```

---

## Scoring Formula

### Technical Skills (35pt)
```
match_ratio      = len(matched) / (len(matched) + len(missing_critical))
critical_penalty = len(missing_critical) × 0.12
effective_ratio  = max(0, match_ratio - critical_penalty)
raw_score        = effective_ratio × 35
```

### Experience (20pt)
```
year_ratio   = min(candidate_years / required_years, 1.2)
year_score   = year_ratio × 0.6
domain_score = domain_relevance × 0.4    # 0.0–1.0 từ LLM
raw_score    = min(year_score + domain_score, 1.0) × 20
```

### Tier
| Tier | Score | Ý nghĩa |
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

# LangSmith — optional nhưng khuyến khích để debug
LANGSMITH_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=hr-cv-scanner
LANGCHAIN_TRACING_V2=true

# ChromaDB — mặc định dùng CWD/data/chroma
CHROMA_DIR=data/chroma

# App
API_BASE_URL=http://localhost:8000/api/v1   # Streamlit gọi FastAPI

# AWS — optional, có fallback lưu local
S3_BUCKET_NAME=
AWS_REGION=ap-southeast-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
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

### Chạy Backend
```powershell
uvicorn api.main:app --reload --port 8000
# Swagger: http://localhost:8000/docs
# Health:  http://localhost:8000/health
```

### Chạy Frontend
```powershell
streamlit run frontend/streamlit_app.py
# UI: http://localhost:8501
```

### Workflow sử dụng
```
1. Mở http://localhost:8501
2. Sidebar: nhập Gemini API key
3. Sidebar: chọn Job (hoặc tạo JD qua API trước)
4. Tab "Scan CVs":
   - Upload CVs (PDF/DOCX)
   - Paste full JD text vào text area
   - Click "🚀 Bắt đầu Scan"
5. Theo dõi progress (auto-refresh mỗi 3s)
6. Tab "Kết quả": xem bảng xếp hạng
7. Tab "Download": tải PDF report
```

### Tạo JD qua API (cần làm trước khi scan)
```bash
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "backend-2025",
    "job_title": "Senior Backend Engineer",
    "requirements": [
      "3+ years Python backend development",
      "FastAPI or Django REST framework",
      "AWS Lambda and S3",
      "PostgreSQL and Redis",
      "Docker and CI/CD"
    ],
    "nice_to_have": ["Kubernetes", "Terraform"]
  }'
```

---

## Tests

```powershell
# Toàn bộ
pytest tests/ -v

# Từng agent
pytest tests/test_cv_parser_v2.py -v    # Agent 1 v2 (markdown, chunking)
pytest tests/test_jd_matcher_v2.py -v   # Agent 2 v2 (JD parse, evidence query)
pytest tests/test_scorer.py -v           # Agent 3
pytest tests/test_report_writer.py -v   # Agent 4
pytest tests/test_orchestrator.py -v    # LangGraph pipeline
pytest tests/test_api.py -v             # FastAPI endpoints

# 1 test cụ thể trong PyCharm: click ▶️ bên cạnh def test_...
```

---

## Quan trọng khi sửa code

1. **Agent 1 không còn dùng LLM** — không cần `api_key` trong `CVParserAgent()`. Nếu cần test với file thật: `agent.parse("cv.pdf")` không cần key.

2. **Agent 2 cần `jd_text`** — khi gọi `match_batch()` phải truyền `jd_text` (full JD text). `GraphState.jd_text` được set từ form upload của Streamlit.

3. **ChromaDB có 2 collections tách biệt:**
   - `job_descriptions`: JD requirements, ingest qua API, persistent
   - `cv_chunks`: CV sections, tạo tự động khi Agent 1 parse, upsert theo `cv_id`

4. **`cv_id` là idempotent** — `cv_{md5(filename)[:12]}`. Cùng filename → cùng `cv_id` → upsert không duplicate. Muốn force re-process: `CVParserAgent(reprocess=True)`.

5. **Scorer nhận `list[ParsedCV]`** thay vì `list[CandidateProfile]`. LLM scoring dùng `candidate.markdown` sections thay vì structured fields.

6. **Thay LLM**: chỉ cần thay `model` param trong constructor. Interface `_call_llm()` giữ nguyên ở tất cả agents.

7. **Scale up**: LangGraph `Send()` API để fan-out parse/match per file song song. Hiện tại sequential.

8. **Production**: thay `scan_jobs` dict bằng DynamoDB/Redis. Set `S3_BUCKET_NAME` để lưu PDF lên S3.

9. **Windows + Python 3.14**: dùng Streamlit, không dùng Chainlit (Chainlit bị `anyio.NoEventLoopError`).

---

## LLM Calls Summary (per candidate)

| Node | LLM Calls | Mô tả |
|---|---|---|
| parse_node | 0 | Không dùng LLM |
| match_node | 1 (cached) + 1 | JD parse (cache) + match analyze |
| score_node | 1 | education + achievements + soft_skills |
| report_node | 0 | Không dùng LLM |
| **Total/candidate** | **~2** | Giảm từ ~4 calls (v1) |
| **Total/batch 50 CVs** | **~101** | 1 JD parse + 50 match + 50 score |