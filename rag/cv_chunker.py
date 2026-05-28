"""
RAG: CV Chunker

Pipeline: Markdown → Chunks → Embeddings → ChromaDB

Chunking strategy (2 bước):
  Bước 1 — Split theo H1 heading (# Experience, # Skills...)
            Mỗi section là 1 chunk ban đầu
            Dùng: MarkdownHeaderTextSplitter

  Bước 2 — Nếu chunk > MAX_CHUNK_CHARS → split đệ quy theo:
            ["\n\n", "\n", ". ", " "] (thứ tự ưu tiên)
            Dùng: RecursiveCharacterTextSplitter

Tại sao MAX_CHUNK_CHARS = 900?
  all-MiniLM-L6-v2 max input = 256 tokens
  Average English: ~4 chars/token → 256 * 4 = ~1024 chars
  Dùng 900 để có buffer an toàn (tránh truncation)

ChromaDB collection: "cv_chunks"
  Metadata per chunk:
    cv_id:        UUID của CV này (dùng để filter khi query)
    file_name:    Tên file gốc
    section:      H1 heading section (e.g. "Experience", "Skills")
    chunk_index:  Thứ tự chunk trong CV (0-based)
    char_count:   Số ký tự của chunk
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.embedder import embed_batch
from rag.ingestor import CHROMA_DIR

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
CV_COLLECTION    = "cv_chunks"
MAX_CHUNK_CHARS  = 900     # ~256 tokens với all-MiniLM-L6-v2
CHUNK_OVERLAP    = 80      # Overlap giữa các sub-chunks để không mất context

# Headers dùng để split bước 1
SPLIT_HEADERS = [
    ("#",  "section"),    # H1 → trường "section" trong metadata
    ("##", "subsection"), # H2 → trường "subsection"
]


@dataclass
class CVChunk:
    """1 chunk của CV sau khi split."""
    text:        str
    cv_id:       str
    file_name:   str
    section:     str   # H1 header (e.g. "Experience")
    subsection:  str   # H2 header nếu có (e.g. "FPT Software")
    chunk_index: int
    char_count:  int   = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class ChunkingResult:
    """Kết quả sau khi chunk + store toàn bộ CV."""
    cv_id:       str
    file_name:   str
    chunk_count: int
    sections:    list[str]   # Danh sách sections đã detect được


# ── Public API ─────────────────────────────────────────────────────────────

def chunk_and_store(
    markdown:  str,
    file_name: str,
    cv_id:     str | None = None,
) -> ChunkingResult:
    """
    Chunk markdown CV và store vào ChromaDB.

    Args:
        markdown:  Markdown string từ cv_to_markdown tool
        file_name: Tên file gốc (dùng làm metadata)
        cv_id:     UUID cho CV này. None = tự sinh UUID mới.
                   Truyền vào nếu muốn overwrite CV cũ (re-process).

    Returns:
        ChunkingResult với cv_id và thống kê
    """
    if not markdown.strip():
        raise ValueError("markdown cannot be empty")

    cv_id = cv_id or str(uuid.uuid4())

    # Bước 1: Split theo heading
    chunks = _split_by_headers(markdown, cv_id, file_name)

    # Bước 2: Sub-split chunks quá dài
    chunks = _split_oversized(chunks)

    if not chunks:
        raise ValueError(f"No chunks produced from {file_name}")

    # Bước 3: Embed + store
    _store_chunks(chunks)

    sections = list(dict.fromkeys(
        c.section for c in chunks if c.section
    ))

    logger.info(
        "Chunked & stored: %s | cv_id=%s | %d chunks | sections=%s",
        file_name, cv_id, len(chunks), sections,
    )

    return ChunkingResult(
        cv_id       = cv_id,
        file_name   = file_name,
        chunk_count = len(chunks),
        sections    = sections,
    )


def delete_cv_chunks(cv_id: str) -> int:
    """
    Xóa toàn bộ chunks của 1 CV khỏi ChromaDB.
    Dùng khi re-process CV.
    """
    client     = _get_client()
    collection = client.get_or_create_collection(CV_COLLECTION)
    results    = collection.get(where={"cv_id": {"$eq": cv_id}})
    ids        = results.get("ids", [])

    if ids:
        collection.delete(ids=ids)
        logger.info("Deleted %d chunks for cv_id=%s", len(ids), cv_id)

    return len(ids)


def query_cv_chunks(
    query_text: str,
    cv_id:      str,
    top_k:      int = 5,
    min_score:  float = 0.0,   # Không filter — lấy tất cả liên quan
) -> list[dict]:
    """
    Query ChromaDB để lấy chunks của 1 CV liên quan đến query_text.

    Args:
        query_text: Skill hoặc requirement cần tìm (từ JD)
        cv_id:      Filter chỉ lấy chunks của CV này
        top_k:      Số chunks tối đa trả về
        min_score:  Score tối thiểu (0.0 = không filter)

    Returns:
        List dicts với keys: text, section, subsection, score, chunk_index
    """
    client     = _get_client()
    collection = client.get_or_create_collection(CV_COLLECTION)

    if collection.count() == 0:
        logger.warning("cv_chunks collection is empty")
        return []

    from rag.embedder import embed_text
    query_vector = embed_text(query_text)

    # Count chunks của cv_id này để không query vượt quá
    cv_chunks_result = collection.get(where={"cv_id": {"$eq": cv_id}})
    cv_chunk_count = len(cv_chunks_result.get("ids", []))

    if cv_chunk_count == 0:
        logger.warning("No chunks found for cv_id=%s", cv_id)
        return []

    n_results = min(top_k, cv_chunk_count)

    results = collection.query(
        query_embeddings = [query_vector],
        n_results        = n_results,
        where            = {"cv_id": {"$eq": cv_id}},
        include          = ["documents", "metadatas", "distances"],
    )

    chunks = []
    if not results["ids"] or not results["ids"][0]:
        return chunks

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1.0 - distance / 2.0, 4)
        if score < min_score:
            continue
        chunks.append({
            "text":        doc,
            "section":     meta.get("section",    ""),
            "subsection":  meta.get("subsection", ""),
            "score":       score,
            "chunk_index": meta.get("chunk_index", 0),
        })

    return chunks


def get_all_cv_chunks(cv_id: str) -> list[dict]:
    """
    Lấy TẤT CẢ chunks của 1 CV (không filter theo query).
    Dùng để Agent 2 có full context.
    """
    client     = _get_client()
    collection = client.get_or_create_collection(CV_COLLECTION)

    results = collection.get(
        where   = {"cv_id": {"$eq": cv_id}},
        include = ["documents", "metadatas"],
    )

    chunks = []
    for doc, meta in zip(
        results.get("documents", []),
        results.get("metadatas",  []),
    ):
        chunks.append({
            "text":        doc,
            "section":     meta.get("section",    ""),
            "subsection":  meta.get("subsection", ""),
            "chunk_index": meta.get("chunk_index", 0),
        })

    # Sort theo chunk_index để đọc theo thứ tự
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks


# ── Private: Chunking ──────────────────────────────────────────────────────

def _split_by_headers(
    markdown:  str,
    cv_id:     str,
    file_name: str,
) -> list[CVChunk]:
    """
    Bước 1: Split Markdown theo H1/H2 headings.
    Mỗi section (Experience, Skills...) thành 1 chunk ban đầu.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on = SPLIT_HEADERS,
        strip_headers       = False,   # Giữ heading trong text để LLM biết context
    )

    docs   = splitter.split_text(markdown)
    chunks = []

    # Nếu không có heading nào → cả CV là 1 chunk
    if not docs:
        chunks.append(CVChunk(
            text       = markdown,
            cv_id      = cv_id,
            file_name  = file_name,
            section    = "Full CV",
            subsection = "",
            chunk_index = 0,
        ))
        return chunks

    for i, doc in enumerate(docs):
        section    = doc.metadata.get("section",    "")
        subsection = doc.metadata.get("subsection", "")

        chunks.append(CVChunk(
            text        = doc.page_content,
            cv_id       = cv_id,
            file_name   = file_name,
            section     = section,
            subsection  = subsection,
            chunk_index = i,
        ))

    return chunks


def _split_oversized(chunks: list[CVChunk]) -> list[CVChunk]:
    """
    Bước 2: Chunk nào > MAX_CHUNK_CHARS → split đệ quy theo:
    ["\n\n", "\n", ". ", " "] — giữ context tốt nhất có thể.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size         = MAX_CHUNK_CHARS,
        chunk_overlap      = CHUNK_OVERLAP,
        separators         = ["\n\n", "\n", ". ", " ", ""],
        length_function    = len,
        is_separator_regex = False,
    )

    result    = []
    global_idx = 0

    for chunk in chunks:
        if len(chunk.text) <= MAX_CHUNK_CHARS:
            # Đủ nhỏ → giữ nguyên, chỉ update index
            result.append(CVChunk(
                text        = chunk.text,
                cv_id       = chunk.cv_id,
                file_name   = chunk.file_name,
                section     = chunk.section,
                subsection  = chunk.subsection,
                chunk_index = global_idx,
            ))
            global_idx += 1
        else:
            # Quá lớn → split
            sub_texts = splitter.split_text(chunk.text)
            logger.debug(
                "Sub-split section '%s': %d chars → %d sub-chunks",
                chunk.section, len(chunk.text), len(sub_texts),
            )
            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue
                result.append(CVChunk(
                    text        = sub_text,
                    cv_id       = chunk.cv_id,
                    file_name   = chunk.file_name,
                    section     = chunk.section,
                    subsection  = chunk.subsection,
                    chunk_index = global_idx,
                ))
                global_idx += 1

    return result


# ── Private: ChromaDB ──────────────────────────────────────────────────────

def _get_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path     = str(CHROMA_DIR),
        settings = Settings(anonymized_telemetry=False),
    )


def _store_chunks(chunks: list[CVChunk]) -> None:
    """Embed tất cả chunks và upsert vào ChromaDB."""
    client     = _get_client()
    collection = client.get_or_create_collection(
        name     = CV_COLLECTION,
        metadata = {"hnsw:space": "cosine"},
    )

    texts     = [c.text for c in chunks]
    vectors   = embed_batch(texts)

    ids       = [f"{c.cv_id}_chunk_{c.chunk_index:04d}" for c in chunks]
    metadatas = [
        {
            "cv_id":       c.cv_id,
            "file_name":   c.file_name,
            "section":     c.section,
            "subsection":  c.subsection,
            "chunk_index": c.chunk_index,
            "char_count":  c.char_count,
        }
        for c in chunks
    ]

    collection.upsert(
        ids        = ids,
        documents  = texts,
        embeddings = vectors,
        metadatas  = metadatas,
    )
    logger.debug("Upserted %d chunks to ChromaDB collection '%s'", len(chunks), CV_COLLECTION)