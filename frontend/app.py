import sys
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import os
from io import BytesIO

import httpx
import chainlit as cl

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

TIER_EMOJI = {
    "strong":   "🟢",
    "good":     "🔵",
    "moderate": "🟡",
    "weak":     "🔴",
}


@cl.on_chat_start
async def on_start():
    jobs = await _fetch_jobs()

    lines = [
        "👋 **Chào mừng đến HR CV Scanner!**\n",
        "Hệ thống multi-agent tự động:",
        "1. Parse CV (PDF/DOCX)",
        "2. Match với Job Description",
        "3. Chấm điểm theo rubric",
        "4. Xuất báo cáo PDF có bảng xếp hạng\n",
    ]

    if jobs:
        lines.append("**Jobs đã setup:**")
        for j in jobs:
            lines.append(f"  - `{j['job_id']}` — {j['job_title']}")
        lines.append("")
    else:
        lines.append("⚠️ Chưa có Job nào. Dùng API `POST /api/v1/jobs` để tạo JD.\n")

    lines += [
        "**Cách dùng:**",
        "1. Gõ `/setjob <job_id> <job_title>`",
        "2. Upload CVs bằng nút 📎 ở góc trái ô chat",
        "3. Gõ `/scan` để bắt đầu\n",
        "Gõ `/help` để xem tất cả commands.",
    ]

    await cl.Message(content="\n".join(lines)).send()

    cl.user_session.set("jobs",              jobs)
    cl.user_session.set("current_job_id",    "")
    cl.user_session.set("current_job_title", "")
    cl.user_session.set("api_key",           os.getenv("GEMINI_API_KEY", ""))
    cl.user_session.set("pending_files",     [])


@cl.on_message
async def on_message(message: cl.Message):
    if message.elements:
        cv_files = [el for el in message.elements if isinstance(el, cl.File)]
        if cv_files:
            await _handle_file_upload(cv_files)
            return

    text = message.content.strip()

    if text.startswith("/help"):
        await _show_help()
    elif text.startswith("/jobs"):
        await _show_jobs()
    elif text.startswith("/setjob"):
        await _set_job(text)
    elif text.startswith("/setkey"):
        await _set_api_key(text)
    elif text.startswith("/scan"):
        await _start_scan_from_session()
    elif text.startswith("/status"):
        await _check_status(text)
    else:
        await cl.Message(
            content="❓ Không hiểu lệnh. Gõ `/help` để xem danh sách.\nĐể upload CV: dùng nút 📎 ở góc trái ô chat."
        ).send()


async def _handle_file_upload(files: list):
    cl.user_session.set("pending_files", files)

    file_list = "\n".join(f"  📄 {f.name}" for f in files)
    job_id    = cl.user_session.get("current_job_id", "")
    job_title = cl.user_session.get("current_job_title", "")
    api_key   = cl.user_session.get("api_key", "")

    issues = []
    if not job_id:
        issues.append("Chưa chọn Job — gõ `/setjob <id> <title>`")
    if not api_key:
        issues.append("Chưa có API key — gõ `/setkey YOUR_KEY`")

    if issues:
        issue_text = "\n".join(f"  • {i}" for i in issues)
        await cl.Message(
            content=f"✅ Đã nhận **{len(files)} files:**\n{file_list}\n\n⚠️ Cần bổ sung:\n{issue_text}\n\nSau đó gõ `/scan` để bắt đầu."
        ).send()
        return

    await cl.Message(
        content=f"✅ Đã nhận **{len(files)} files:**\n{file_list}\n\n🎯 Job: **{job_title}** (`{job_id}`)\n\nBắt đầu scan không?",
        actions=[
            cl.Action(name="confirm_scan", label="🚀 Bắt đầu Scan", value="yes"),
            cl.Action(name="cancel_scan",  label="❌ Huỷ",           value="no"),
        ],
    ).send()


@cl.action_callback("confirm_scan")
async def on_confirm_scan(action: cl.Action):
    await action.remove()
    await _start_scan_from_session()


@cl.action_callback("cancel_scan")
async def on_cancel_scan(action: cl.Action):
    await action.remove()
    cl.user_session.set("pending_files", [])
    await cl.Message(content="❌ Đã huỷ.").send()


async def _start_scan_from_session():
    files     = cl.user_session.get("pending_files", [])
    job_id    = cl.user_session.get("current_job_id",    "")
    job_title = cl.user_session.get("current_job_title", "")
    api_key   = cl.user_session.get("api_key",           "")

    errors = []
    if not files:
        errors.append("Chưa upload CV files (dùng nút 📎).")
    if not job_id:
        errors.append("Chưa chọn Job ID (`/setjob`).")
    if not api_key:
        errors.append("Chưa có API key (`/setkey`).")

    if errors:
        error_text = "\n".join(f"  • {e}" for e in errors)
        await cl.Message(content=f"⚠️ Cần bổ sung:\n{error_text}").send()
        return

    status_msg = await cl.Message(
        content=f"⏳ Đang upload {len(files)} files lên backend..."
    ).send()

    run_id = await _upload_and_start(files, job_id, job_title, api_key)

    if not run_id:
        await status_msg.update(
            content="❌ Upload thất bại. Kiểm tra FastAPI đang chạy ở port 8000."
        )
        return

    cl.user_session.set("pending_files", [])
    await status_msg.update(
        content=f"🚀 Pipeline started! Run ID: `{run_id}`\nĐang theo dõi tiến trình..."
    )

    await _stream_progress(run_id, status_msg)


async def _upload_and_start(files, job_id, job_title, api_key):
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            form_files = []
            for f in files:
                if hasattr(f, "path") and f.path:
                    with open(f.path, "rb") as fh:
                        content = fh.read()
                elif hasattr(f, "content") and f.content:
                    content = f.content
                else:
                    continue

                mime = (
                    "application/pdf"
                    if f.name.lower().endswith(".pdf")
                    else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                form_files.append(("files", (f.name, BytesIO(content), mime)))

            if not form_files:
                return None

            r = await client.post(
                f"{API_BASE}/scan/start",
                files=form_files,
                data={"job_id": job_id, "job_title": job_title, "api_key": api_key},
            )
            r.raise_for_status()
            return r.json()["run_id"]

    except Exception as e:
        print(f"Upload error: {e}")
        return None


async def _stream_progress(run_id, status_msg):
    stream_url = f"{API_BASE}/scan/stream/{run_id}"
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", stream_url) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue

                    data     = json.loads(line[5:].strip())
                    status   = data.get("status", "")
                    step     = data.get("step",   "")
                    progress = data.get("progress", 0)

                    bar = _progress_bar(progress)
                    await status_msg.update(
                        content=f"**Pipeline: {status.upper()}**\n{bar}\n\n📍 {step}"
                    )

                    if status in {"completed", "failed"}:
                        if status == "completed":
                            await _show_results(run_id, status_msg)
                        else:
                            err = data.get("error", "")
                            await status_msg.update(content=f"❌ **Thất bại**\n\n{err}")
                        break

    except Exception as e:
        await status_msg.update(
            content=f"⚠️ Mất kết nối SSE: {e}\n\nDùng `/status {run_id}` để kiểm tra."
        )


async def _show_results(run_id, status_msg):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{API_BASE}/reports/run/{run_id}/shortlist")
            r.raise_for_status()
            data = r.json()

        shortlist = data.get("shortlist", [])
        job_title = data.get("job_title", "")

        lines = [f"✅ **Done! — {job_title}**\n", f"**Top {len(shortlist)} Candidates:**\n"]
        for c in shortlist:
            emoji = TIER_EMOJI.get(c["tier"], "⚪")
            line  = f"{emoji} **#{c['rank']} {c['name']}** — {c['score']}/100"
            if c.get("strengths"):
                line += f"\n  ✓ {c['strengths'][0]}"
            if c.get("concerns"):
                line += f"\n  △ {c['concerns'][0]}"
            lines.append(line)

        await status_msg.update(content="\n".join(lines))

        await cl.Message(
            content="📄 **Report sẵn sàng:**",
            actions=[
                cl.Action(name="download_pdf", label="⬇️ Download PDF Report", value=run_id)
            ],
        ).send()

    except Exception as e:
        await status_msg.update(content=f"✅ Pipeline xong! Lỗi khi lấy shortlist: {e}")


@cl.action_callback("download_pdf")
async def on_download_pdf(action: cl.Action):
    run_id  = action.value
    pdf_url = f"{API_BASE}/reports/run/{run_id}/download"
    await cl.Message(content=f"📥 **Download PDF:**\n[Click để download]({pdf_url})").send()


async def _show_help():
    lines = [
        "**📖 Commands:**\n",
        "`/jobs`                        — Xem Jobs đã setup",
        "`/setjob <job_id> <job_title>` — Chọn Job để scan",
        "`/setkey <api_key>`            — Set Gemini API key",
        "`/scan`                        — Bắt đầu scan",
        "`/status <run_id>`             — Kiểm tra tiến trình\n",
        "**Upload CVs:** Dùng nút 📎 ở góc trái ô chat\n",
        "**Ví dụ flow:**",
        "```",
        "/setjob backend-2025 Senior Backend Engineer",
        "[upload CVs bằng nút 📎]",
        "/scan",
        "```",
    ]
    await cl.Message(content="\n".join(lines)).send()


async def _show_jobs():
    jobs = await _fetch_jobs()
    if not jobs:
        await cl.Message(content="📭 Chưa có Job nào. Xem README để tạo JD.").send()
        return

    lines = ["**📋 Jobs đã setup:**\n"]
    for j in jobs:
        lines.append(f"  • `{j['job_id']}` — {j['job_title']}")
    lines.append("\n_Dùng `/setjob <job_id> <title>` để chọn._")
    await cl.Message(content="\n".join(lines)).send()


async def _set_job(text):
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        await cl.Message(
            content="⚠️ Cú pháp: `/setjob <job_id> <job_title>`\nVí dụ: `/setjob backend-2025 Senior Backend Engineer`"
        ).send()
        return

    job_id    = parts[1]
    job_title = parts[2].strip("\"'")
    cl.user_session.set("current_job_id",    job_id)
    cl.user_session.set("current_job_title", job_title)
    await cl.Message(
        content=f"✅ Đã chọn Job:\n  • ID: `{job_id}`\n  • Title: **{job_title}**\n\nBây giờ upload CVs bằng nút 📎."
    ).send()


async def _set_api_key(text):
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await cl.Message(content="⚠️ Cú pháp: `/setkey YOUR_GEMINI_API_KEY`").send()
        return

    key    = parts[1].strip()
    masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
    cl.user_session.set("api_key", key)
    await cl.Message(content=f"✅ API key đã set: `{masked}`").send()


async def _check_status(text):
    parts = text.split()
    if len(parts) < 2:
        await cl.Message(content="⚠️ Cú pháp: `/status <run_id>`").send()
        return

    run_id = parts[1].upper()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{API_BASE}/scan/status/{run_id}")
            if r.status_code == 404:
                await cl.Message(content=f"❌ Run `{run_id}` không tìm thấy.").send()
                return
            data = r.json()

        bar     = _progress_bar(data.get("progress", 0))
        content = f"**Status: `{run_id}`**\n\nStatus: `{data['status']}`\n{bar}\nStep: {data.get('step', '')}"
        if data.get("report_id"):
            content += f"\nReport ID: `{data['report_id']}`"
        if data.get("error"):
            content += f"\n❌ Error: {data['error']}"

        await cl.Message(content=content).send()

    except Exception as e:
        await cl.Message(content=f"❌ Lỗi: {e}").send()


async def _fetch_jobs():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{API_BASE}/jobs/")
            r.raise_for_status()
            return r.json().get("jobs", [])
    except Exception:
        return []


def _progress_bar(progress):
    filled = int(progress * 20)
    empty  = 20 - filled
    pct    = int(progress * 100)
    return f"`[{'█' * filled}{'░' * empty}] {pct}%`"