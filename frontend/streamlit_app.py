"""
Streamlit Frontend — HR CV Scanner v2

Thay đổi so với v1:
  + Thêm JD text area (Agent 2 v2 cần full JD text để parse)
  + Fix double sleep bug trong progress tracking
  + Sidebar gọn hơn, validation rõ hơn
"""
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

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "job_id":    "",
        "job_title": "",
        "jd_text":   "",
        "api_key":   os.getenv("GEMINI_API_KEY", ""),
        "run_id":    None,
        "status":    None,
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


def _progress_bar(p: float) -> str:
    filled = int(p * 20)
    return f"[{'█' * filled}{'░' * (20 - filled)}] {int(p * 100)}%"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Cấu hình")

    # Gemini API Key
    st.subheader("🔑 Gemini API Key")
    api_key = st.text_input(
        "API Key",
        value = st.session_state.api_key,
        type  = "password",
        help  = "Lấy miễn phí tại aistudio.google.com/apikey",
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
        st.session_state.job_id    = st.text_input("Job ID",    value=st.session_state.job_id)
        st.session_state.job_title = st.text_input("Job Title", value=st.session_state.job_title)

    st.divider()

    # Backend status
    st.subheader("🔌 Backend")
    try:
        h = httpx.get("http://localhost:8000/health", timeout=2)
        d = h.json()
        if h.status_code == 200:
            st.success("FastAPI: Running ✅")
            active = d.get("active_scans", 0)
            if active:
                st.info(f"Đang xử lý: {active} scan(s)")
        else:
            st.error("FastAPI: Error ❌")
    except Exception:
        st.error("FastAPI: Offline ❌")
        st.code("uvicorn api.main:app --reload --port 8000")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📋 HR CV Scanner")
st.caption("Multi-agent AI: Parse → Match → Score → Report")

tab1, tab2, tab3 = st.tabs(["🚀 Scan CVs", "📊 Kết quả", "📥 Download"])


# ─── Tab 1: Scan ──────────────────────────────────────────────────────────────
with tab1:

    col_left, col_right = st.columns([1, 1], gap="large")

    # Cột trái: Upload CVs
    with col_left:
        st.subheader("📤 Upload CVs")
        uploaded_files = st.file_uploader(
            "Chọn CV files (PDF/DOCX)",
            type                  = ["pdf", "docx", "doc"],
            accept_multiple_files = True,
            help                  = "Hỗ trợ PDF, DOCX, DOC",
        )
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) đã chọn")
            for f in uploaded_files:
                st.caption(f"📄 {f.name}")

    # Cột phải: JD Text
    with col_right:
        st.subheader("📋 Job Description")
        st.caption("Paste full JD text — Agent 2 sẽ tự phân tích thành các trường yêu cầu")
        jd_text = st.text_area(
            "Full JD Text",
            value       = st.session_state.jd_text,
            height      = 250,
            placeholder = (
                "Senior Backend Engineer\n\n"
                "Requirements:\n"
                "- 3+ years Python backend development\n"
                "- FastAPI or Django\n"
                "- AWS Lambda and S3\n"
                "- PostgreSQL and Redis\n"
                "- Docker and CI/CD\n\n"
                "Nice to have:\n"
                "- Kubernetes\n"
                "- Terraform"
            ),
            label_visibility = "collapsed",
        )
        st.session_state.jd_text = jd_text
        if jd_text:
            word_count = len(jd_text.split())
            st.caption(f"📝 {word_count} từ")

    st.divider()

    # Validation
    issues = []
    if not st.session_state.api_key:
        issues.append("⚠️ Chưa có Gemini API Key (nhập ở sidebar)")
    if not st.session_state.job_id:
        issues.append("⚠️ Chưa chọn Job (chọn ở sidebar)")
    if not jd_text or len(jd_text.strip()) < 50:
        issues.append("⚠️ JD text quá ngắn — paste đầy đủ JD để Agent 2 phân tích chính xác")
    if not uploaded_files:
        issues.append("⚠️ Chưa upload CV files")

    for issue in issues:
        st.warning(issue)

    # Nút scan
    can_scan = len(issues) == 0
    if st.button(
        "🚀 Bắt đầu Scan",
        disabled            = not can_scan,
        type                = "primary",
        use_container_width = True,
    ):
        with st.spinner(f"Đang upload {len(uploaded_files)} files..."):
            form_files = []
            for f in uploaded_files:
                mime = (
                    "application/pdf"
                    if f.name.lower().endswith(".pdf")
                    else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                form_files.append(("files", (f.name, BytesIO(f.read()), mime)))

            result = None
            try:
                r = httpx.post(
                    f"{API_BASE}/scan/start",
                    files   = form_files,
                    data    = {
                        "job_id":    st.session_state.job_id,
                        "job_title": st.session_state.job_title,
                        "jd_text":   st.session_state.jd_text,
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
        st.subheader(f"⏳ Đang chạy — `{run_id}`")

        # 1 lần fetch mỗi rerun (không block)
        data = _get(f"/scan/status/{run_id}")
        if not data:
            st.error("Không kết nối được backend.")
        else:
            progress = float(data.get("progress", 0))
            step     = data.get("step",   "")
            status   = data.get("status", "running")
            error    = data.get("error",  "")

            st.progress(progress, text=f"**{status.upper()}** — {_progress_bar(progress)}")
            st.info(f"📍 {step}")

            if status == "completed":
                st.session_state.status = "completed"
                st.balloons()
                st.success("✅ Xong! Xem kết quả ở tab **Kết quả**.")
                st.rerun()

            elif status == "failed":
                st.session_state.status = "failed"
                st.error(f"❌ Thất bại: {error}")

            else:
                # Vẫn đang chạy → rerun sau 3s
                time.sleep(3)
                st.rerun()


# ─── Tab 2: Kết quả ───────────────────────────────────────────────────────────
with tab2:
    run_id = st.session_state.run_id

    if not run_id:
        st.info("Chạy scan ở tab **Scan CVs** trước.")
    else:
        st.subheader(f"Kết quả — Run `{run_id}`")

        report_data = _get(f"/reports/run/{run_id}")

        if report_data is None:
            status_data = _get(f"/scan/status/{run_id}")
            if status_data:
                pct = float(status_data.get("progress", 0))
                st.progress(pct, text=f"Status: `{status_data['status']}`")
                st.info(status_data.get("step", ""))
        else:
            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng ứng viên",  report_data["total_candidates"])
            c2.metric("Shortlist",       report_data["shortlist_count"])
            c3.metric("Report ID",       report_data["report_id"])
            c4.metric("Job",             report_data["job_title"][:18])

            st.divider()

            # LangSmith link
            langsmith_key = os.getenv("LANGSMITH_API_KEY", "")
            if langsmith_key:
                project = os.getenv("LANGCHAIN_PROJECT", "hr-cv-scanner")
                st.info(
                    f"🔍 [Xem chi tiết agent trace trên LangSmith]"
                    f"(https://smith.langchain.com/o/default/projects/{project})"
                )

            # Summary
            with st.expander("📋 Summary", expanded=True):
                st.markdown(report_data.get("summary_text", ""))

            st.divider()

            # Bảng xếp hạng
            st.subheader("🏆 Bảng xếp hạng")
            shortlist = report_data.get("shortlist", [])

            if shortlist:
                for c in shortlist:
                    emoji = TIER_COLOR.get(c.get("tier", ""), "⚪")
                    with st.container(border=True):
                        col_rank, col_name, col_score, col_tier, col_rec = \
                            st.columns([1, 3, 2, 2, 4])
                        col_rank.markdown(f"**#{c['rank']}**")
                        col_name.markdown(f"**{c['name']}**")
                        col_score.markdown(f"`{c['total_score']}/100`")
                        col_tier.markdown(f"{emoji} {c.get('tier','').upper()}")
                        col_rec.caption(c.get("recommendation", "") or "")
            else:
                st.warning("Không có dữ liệu shortlist.")


# ─── Tab 3: Download ──────────────────────────────────────────────────────────
with tab3:
    run_id = st.session_state.run_id

    if not run_id:
        st.info("Chạy scan ở tab **Scan CVs** trước.")
    elif st.session_state.status != "completed":
        st.info("⏳ Đợi pipeline hoàn thành trước khi download.")
    else:
        st.subheader("📥 Download PDF Report")
        try:
            r = httpx.get(
                f"{API_BASE}/reports/run/{run_id}/download",
                timeout = 30,
            )
            r.raise_for_status()
            pdf_bytes = r.content

            st.download_button(
                label               = "⬇️ Download PDF Report",
                data                = pdf_bytes,
                file_name           = f"cv_report_{run_id}.pdf",
                mime                = "application/pdf",
                type                = "primary",
                use_container_width = True,
            )
            st.caption(f"Size: {len(pdf_bytes) / 1024:.1f} KB")

        except Exception as e:
            st.error(f"Không lấy được PDF: {e}")
            st.code(f"{API_BASE}/reports/run/{run_id}/download")