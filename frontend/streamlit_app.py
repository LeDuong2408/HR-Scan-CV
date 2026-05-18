"""
Streamlit Frontend — HR CV Scanner

Chạy: streamlit run frontend/streamlit_app.py --server.port 8001

Không cần Chainlit — Streamlit ổn định hơn trên Windows.
"""
import json
import os
import time
from io import BytesIO

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

TIER_COLOR = {
    "strong":   "🟢",
    "good":     "🔵",
    "moderate": "🟡",
    "weak":     "🔴",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "HR CV Scanner",
    page_icon  = "📋",
    layout     = "wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "job_id":    "",
        "job_title": "",
        "api_key":   os.getenv("GEMINI_API_KEY", ""),
        "run_id":    None,
        "status":    None,
        "report":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get(path: str, timeout: int = 10) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _post(path: str, **kwargs) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _progress_bar_text(p: float) -> str:
    filled = int(p * 20)
    return f"[{'█' * filled}{'░' * (20 - filled)}] {int(p * 100)}%"


# ── Sidebar: Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Cấu hình")

    # API key
    st.subheader("🔑 Gemini API Key")
    api_key = st.text_input(
        "API Key",
        value    = st.session_state.api_key,
        type     = "password",
        help     = "Lấy miễn phí tại aistudio.google.com/apikey",
    )
    st.session_state.api_key = api_key

    st.divider()

    # Job selection
    st.subheader("🎯 Chọn Job")
    jobs_data = _get("/jobs/")
    jobs      = jobs_data.get("jobs", []) if jobs_data else []

    if jobs:
        job_options = {f"{j['job_id']} — {j['job_title']}": j for j in jobs}
        selected    = st.selectbox("Job Description", list(job_options.keys()))
        if selected:
            j = job_options[selected]
            st.session_state.job_id    = j["job_id"]
            st.session_state.job_title = j["job_title"]
    else:
        st.warning("Chưa có Job nào. Tạo qua API trước.")
        st.code(
            'curl -X POST http://localhost:8000/api/v1/jobs/ \\\n'
            '  -H "Content-Type: application/json" \\\n'
            '  -d \'{"job_id":"backend-2025","job_title":"Senior Backend",'
            '"requirements":["Python 3+ years","FastAPI experience"]}\'',
            language="bash",
        )
        # Cho phép nhập tay nếu chưa có
        st.session_state.job_id    = st.text_input("Job ID (nhập tay)", value=st.session_state.job_id)
        st.session_state.job_title = st.text_input("Job Title",         value=st.session_state.job_title)

    st.divider()

    # Backend status
    st.subheader("🔌 Backend Status")
    try:
        h = httpx.get("http://localhost:8000/health", timeout=2)
        if h.status_code == 200:
            st.success("FastAPI: Running ✅")
        else:
            st.error("FastAPI: Error ❌")
    except Exception:
        st.error("FastAPI: Offline ❌\nChạy: `uvicorn api.main:app --reload`")


# ── Main: Upload và Scan ──────────────────────────────────────────────────────
st.title("📋 HR CV Scanner")
st.caption("Multi-agent AI system: Parse → Match → Score → Report")

tab1, tab2, tab3 = st.tabs(["🚀 Scan CVs", "📊 Kết quả", "📥 Download Report"])

# ─── Tab 1: Scan ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload CV Files")

    uploaded_files = st.file_uploader(
        "Chọn CV files (PDF/DOCX)",
        type            = ["pdf", "docx", "doc"],
        accept_multiple_files = True,
        help            = "Kéo thả nhiều files cùng lúc",
    )

    if uploaded_files:
        st.info(f"📄 Đã chọn {len(uploaded_files)} files: " + ", ".join(f.name for f in uploaded_files))

    st.divider()

    # Validate trước khi scan
    ready = True
    if not st.session_state.api_key:
        st.warning("⚠️ Chưa có Gemini API Key — nhập ở sidebar")
        ready = False
    if not st.session_state.job_id:
        st.warning("⚠️ Chưa chọn Job — chọn ở sidebar")
        ready = False
    if not uploaded_files:
        st.warning("⚠️ Chưa upload CV files")
        ready = False

    # Nút Scan
    if st.button(
        "🚀 Bắt đầu Scan",
        disabled = not ready,
        type     = "primary",
        use_container_width = True,
    ):
        with st.spinner("Đang upload files..."):
            form_files = []
            for f in uploaded_files:
                mime = "application/pdf" if f.name.endswith(".pdf") else \
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                form_files.append(("files", (f.name, BytesIO(f.read()), mime)))

            result = None
            try:
                r = httpx.post(
                    f"{API_BASE}/scan/start",
                    files   = form_files,
                    data    = {
                        "job_id":    st.session_state.job_id,
                        "job_title": st.session_state.job_title,
                        "api_key":   st.session_state.api_key,
                    },
                    timeout = 60,
                )
                r.raise_for_status()
                result = r.json()
            except Exception as e:
                st.error(f"❌ Upload thất bại: {e}")

        if result:
            st.session_state.run_id = result["run_id"]
            st.session_state.status = "running"
            st.success(f"✅ Pipeline started! Run ID: `{result['run_id']}`")
            st.rerun()

    # ── Progress tracking ──
    if st.session_state.run_id and st.session_state.status == "running":
        st.divider()
        run_id = st.session_state.run_id
        st.subheader(f"⏳ Pipeline đang chạy — `{run_id}`")

        progress_placeholder = st.empty()
        step_placeholder     = st.empty()

        # Lấy status hiện tại (1 lần mỗi rerun)
        data = _get(f"/scan/status/{run_id}")

        if not data:
            st.error("Không kết nối được backend. Kiểm tra FastAPI đang chạy.")
        else:
            progress = data.get("progress", 0)
            step     = data.get("step",     "")
            status   = data.get("status",   "running")
            error    = data.get("error",    "")

            progress_placeholder.progress(
                float(progress),
                text=f"**{status.upper()}** — {_progress_bar_text(progress)}",
            )
            step_placeholder.info(f"📍 {step}")

            if status == "completed":
                st.session_state.status = "completed"
                st.balloons()
                st.success("✅ Pipeline hoàn thành! Xem kết quả ở tab **Kết quả**.")
                st.rerun()

            elif status == "failed":
                st.session_state.status = "failed"
                st.error(f"❌ Pipeline thất bại: {error}")

            else:
                # Vẫn đang chạy → auto-refresh sau 3 giây
                time.sleep(3)
                st.rerun()

            time.sleep(2)  # Poll mỗi 2 giây


# ─── Tab 2: Kết quả ───────────────────────────────────────────────────────────
with tab2:
    run_id = st.session_state.run_id

    if not run_id:
        st.info("Chạy scan ở tab **Scan CVs** trước.")
    else:
        st.subheader(f"Kết quả — Run `{run_id}`")

        # Lấy report
        report_data = _get(f"/reports/run/{run_id}")

        if report_data is None:
            # Có thể chưa xong
            data = _get(f"/scan/status/{run_id}")
            if data:
                st.info(f"Status: `{data['status']}` — {data.get('step','')}")
                st.progress(data.get("progress", 0))
        else:
            # Summary cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tổng ứng viên",  report_data["total_candidates"])
            col2.metric("Shortlist",       report_data["shortlist_count"])
            col3.metric("Report ID",       report_data["report_id"])
            col4.metric("Job",             report_data["job_title"][:20])

            st.divider()

            # Summary text
            with st.expander("📋 Summary", expanded=True):
                st.markdown(report_data.get("summary_text", ""))

            # Shortlist table
            st.subheader("🏆 Bảng xếp hạng")
            shortlist = report_data.get("shortlist", [])

            if shortlist:
                for c in shortlist:
                    emoji = TIER_COLOR.get(c["tier"], "⚪")
                    with st.container(border=True):
                        col_r, col_n, col_s, col_t, col_rec = st.columns([1, 3, 2, 2, 4])
                        col_r.markdown(f"**#{c['rank']}**")
                        col_n.markdown(f"**{c['name']}**")
                        col_s.markdown(f"`{c['total_score']}/100`")
                        col_t.markdown(f"{emoji} {c['tier'].upper()}")
                        col_rec.markdown(c.get("recommendation", "") or "")
            else:
                st.warning("Không có dữ liệu shortlist.")


# ─── Tab 3: Download ──────────────────────────────────────────────────────────
with tab3:
    run_id = st.session_state.run_id

    if not run_id:
        st.info("Chạy scan ở tab **Scan CVs** trước.")
    elif st.session_state.status != "completed":
        st.info("Đợi pipeline hoàn thành.")
    else:
        st.subheader("📥 Download PDF Report")

        # Lấy PDF binary
        try:
            r = httpx.get(
                f"{API_BASE}/reports/run/{run_id}/download",
                timeout=30,
            )
            r.raise_for_status()
            pdf_bytes = r.content

            st.download_button(
                label    = "⬇️ Download PDF Report",
                data     = pdf_bytes,
                file_name = f"cv_report_{run_id}.pdf",
                mime     = "application/pdf",
                type     = "primary",
                use_container_width = True,
            )

            st.caption(f"Size: {len(pdf_bytes) / 1024:.1f} KB")

        except Exception as e:
            st.error(f"Không lấy được PDF: {e}")
            st.info(
                f"Thử download trực tiếp:\n"
                f"`{API_BASE}/reports/run/{run_id}/download`"
            )