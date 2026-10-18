from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyCookie
import httpx
import json
import logging
import time
import csv
import os
import html
import secrets
from io import StringIO
from collections import defaultdict, deque
from dotenv import load_dotenv
from urllib.parse import quote


# 加载项目根目录 .env 配置文件
load_dotenv()

# ============ 读取环境变量配置，带默认兜底 ============
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE")
MODEL_TEXT = os.getenv("MODEL_TEXT")
MODEL_VISION = os.getenv("MODEL_VISION")

LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", 8000))
TIMEOUT = int(os.getenv("TIMEOUT", 120))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", 6))
WHITE_LIST = json.loads(os.getenv("WHITE_LIST", '["127.0.0.1"]'))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY")

# 请求安全防护（从.env读取，兜底默认值）
MAX_REQUEST_BODY_SIZE = int(os.getenv("MAX_REQUEST_BODY_SIZE", 10 * 1024 * 1024))
MAX_USER_TEXT_LENGTH = int(os.getenv("MAX_USER_TEXT_LENGTH", 6000))

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="机房内网AI助教网关")
app.state.max_request_body_size = MAX_REQUEST_BODY_SIZE

# ================= 简易内存Session存储（管理员后台） =================
admin_cookie_scheme = APIKeyCookie(name="admin_session", auto_error=False)
admin_sessions = dict()  # session_id : {"expire":时间戳}
SESSION_EXPIRE_SEC = 3600  # 会话有效期 1小时


def generate_session_id():
    return secrets.token_hex(32)


async def require_admin_session(cookie=Depends(admin_cookie_scheme)):
    if not cookie:
        raise HTTPException(status_code=401, detail="未登录")
    sess = admin_sessions.get(cookie)
    if not sess or time.time() > sess["expire"]:
        admin_sessions.pop(cookie, None)
        raise HTTPException(status_code=401, detail="会话过期，请重新登录")
    return True


# ================= 业务内存存储 =================
ip_req_records = defaultdict(deque)
student_stats = defaultdict(lambda: {
    "total": 0,
    "last_time": 0,
    "last_msg": "",
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
})
black_ip_set = set()


def check_rate_limit(client_ip: str) -> tuple[bool, str]:
    """IP限流&黑名单校验"""
    if client_ip in black_ip_set:
        return False, "该IP已被管理员限制访问，请联系老师"
    if client_ip in WHITE_LIST:
        return True, ""
    now = time.time()
    queue = ip_req_records[client_ip]
    # 清理60秒以外过期请求记录
    while queue and queue[0] <= now - 60:
        queue.popleft()
    if len(queue) >= RATE_LIMIT_PER_MIN:
        return False, f"请求限流：每台电脑每分钟最多{RATE_LIMIT_PER_MIN}次提问，请稍后重试！"
    queue.append(now)
    return True, ""


async def stream_generator(payload: dict, client_ip: str):
    """向上游LLM转发SSE流式"""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    user_msg = ""
    try:
        for msg in payload["messages"]:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    txt_parts = [c["text"] for c in content if c["type"] == "text"]
                    user_msg = " ".join(txt_parts)
                else:
                    user_msg = str(content)
                break
        student_stats[client_ip]["total"] += 1
        student_stats[client_ip]["last_time"] = time.time()
        student_stats[client_ip]["last_msg"] = user_msg[:120]
    except Exception:
        pass

    api_url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream("POST", api_url, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    err_text = await resp.aread()
                    yield f"data: {json.dumps({'error': f'上游API错误{resp.status_code}:{err_text.decode()}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    parts = buffer.split("\n\n")
                    buffer = parts.pop()
                    for piece in parts:
                        if not piece.startswith("data: "):
                            continue
                        data_str = piece[6:]
                        if data_str.strip() == "[DONE]":
                            continue
                        try:
                            j = json.loads(data_str)
                            yield f"data: {data_str}\n\n"
                            # 统计token（部分国产模型流式不返回usage）
                            if "usage" in j:
                                usage = j["usage"]
                                pt = usage.get("prompt_tokens", 0)
                                ct = usage.get("completion_tokens", 0)
                                tt = usage.get("total_tokens", 0)
                                student_stats[client_ip]["prompt_tokens"] += pt
                                student_stats[client_ip]["completion_tokens"] += ct
                                student_stats[client_ip]["total_tokens"] += tt
                        except Exception:
                            continue
    except Exception as e:
        yield f"data: {json.dumps({'error': f'网关异常：{str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(request: Request):
    # 请求体大小校验
    body_size = int(request.headers.get("content-length", 0))
    if body_size > MAX_REQUEST_BODY_SIZE:
        async def err_stream():
            yield f"data: {json.dumps({'error': '请求体过大，超出限制'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    client_ip = request.client.host
    allow, msg = check_rate_limit(client_ip)
    if not allow:
        async def error_stream():
            yield f"data: {json.dumps({'error': msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    body = await request.json()

    # 单条文本长度校验
    for m in body.get("messages", []):
        cnt = m.get("content")
        if isinstance(cnt, list):
            for item in cnt:
                if item.get("type") == "text" and len(item.get("text", "")) > MAX_USER_TEXT_LENGTH:
                    async def err():
                        yield f"data: {json.dumps({'error': '输入文本过长'})}\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(err(), media_type="text/event-stream")
        elif isinstance(cnt, str) and len(cnt) > MAX_USER_TEXT_LENGTH:
            async def err():
                yield f"data: {json.dumps({'error': '输入文本过长'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(err(), media_type="text/event-stream")

    body["stream"] = True
    has_image = False
    for m in body.get("messages", []):
        cnt = m.get("content")
        if isinstance(cnt, list):
            for item in cnt:
                if item.get("type") == "image_url":
                    has_image = True
                    break
    if has_image:
        body["model"] = MODEL_VISION
    else:
        body["model"] = MODEL_TEXT

    return StreamingResponse(
        stream_generator(body, client_ip),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# ================= 管理员后台接口 =================
@app.post("/api/admin/login")
async def admin_login(req: Request):
    j = await req.json()
    pwd = j.get("pwd", "").strip()
    if pwd != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="密码错误")
    sid = generate_session_id()
    expire = time.time() + SESSION_EXPIRE_SEC
    admin_sessions[sid] = {"expire": expire}
    from fastapi.responses import JSONResponse
    resp = JSONResponse({"ok": True, "msg": "登录成功"})
    resp.set_cookie(key="admin_session", value=sid, httponly=True, max_age=SESSION_EXPIRE_SEC)
    return resp


@app.get("/admin")
async def admin_page():
    return FileResponse("admin.html")


@app.get("/api/admin/stats", dependencies=[Depends(require_admin_session)])
async def get_stats():
    result = []
    now = time.time()
    for ip, data in student_stats.items():
        diff_sec = int(now - data["last_time"])
        if data["last_time"] == 0:
            last_str = "暂无记录"
        else:
            last_str = f"{diff_sec}秒前" if diff_sec < 60 else f"{diff_sec // 60}分{diff_sec % 60}秒前"
        queue = ip_req_records[ip]
        curr_in_min = len(queue)
        is_limited = curr_in_min >= RATE_LIMIT_PER_MIN
        is_black = ip in black_ip_set
        result.append({
            "ip": ip,
            "total": data["total"],
            "last_active": last_str,
            "last_msg": html.escape(data["last_msg"]),
            "current_min_req": curr_in_min,
            "is_limited": is_limited,
            "is_black": is_black,
            "prompt_tokens": data["prompt_tokens"],
            "completion_tokens": data["completion_tokens"],
            "total_tokens": data["total_tokens"]
        })
    result.sort(key=lambda x: x["total"], reverse=True)
    global_total_token = sum(r["total_tokens"] for r in result)
    return {"list": result, "max_per_min": RATE_LIMIT_PER_MIN, "global_total_token": global_total_token}


from urllib.parse import quote

@app.get("/api/admin/export-csv", dependencies=[Depends(require_admin_session)])
async def export_csv():
    output = StringIO()
    output.write("\ufeff")  # BOM头，解决Windows Excel中文乱码
    writer = csv.writer(output)
    writer.writerow([
        "学生IP",
        "累计提问次数",
        "输入Token",
        "输出Token",
        "总Token",
        "最近活跃",
        "限流状态",
        "拉黑状态",
        "最近提问预览"
    ])
    now = time.time()
    for ip, data in student_stats.items():
        diff_sec = int(now - data["last_time"])
        if data["last_time"] == 0:
            last_str = "暂无记录"
        else:
            last_str = f"{diff_sec}秒前" if diff_sec < 60 else f"{diff_sec // 60}分{diff_sec % 60}秒前"
        queue = ip_req_records[ip]
        curr_in_min = len(queue)
        is_limited = curr_in_min >= RATE_LIMIT_PER_MIN
        is_black = ip in black_ip_set
        writer.writerow([
            ip,
            data["total"],
            data["prompt_tokens"],
            data["completion_tokens"],
            data["total_tokens"],
            last_str,
            "已限流" if is_limited else "正常",
            "已拉黑" if is_black else "正常",
            data["last_msg"]
        ])
    output.seek(0)
    csv_content = output.getvalue()

    filename = "ai_student_report.csv"
    encoded_name = quote(filename, encoding="utf-8")
    headers = {
        # 注意！这里是普通英文减号 - 不是软连字符
        "Content-Disposition": f'attachment; filename*=utf-8\'\'{encoded_name}'
    }
    return PlainTextResponse(
        csv_content,
        media_type="text/csv",
        headers=headers
    )


@app.post("/api/admin/reset-limit", dependencies=[Depends(require_admin_session)])
async def reset_limit(ip: str):
    if ip in WHITE_LIST:
        return {"ok": False, "msg": "白名单教师IP无需重置"}
    ip_req_records[ip].clear()
    return {"ok": True, "msg": f"已重置IP {ip} 的限流计数"}


@app.post("/api/admin/toggle-black", dependencies=[Depends(require_admin_session)])
async def toggle_black(ip: str):
    if ip in WHITE_LIST:
        return {"ok": False, "msg": "不能拉黑白名单教师IP"}
    if ip in black_ip_set:
        black_ip_set.remove(ip)
        return {"ok": True, "msg": f"已解除IP {ip} 的访问限制"}
    else:
        black_ip_set.add(ip)
        return {"ok": True, "msg": f"已拉黑IP {ip}，该学生无法继续提问"}


# ⚠️静态文件挂载必须放在所有API路由之后！
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=LISTEN_HOST, port=LISTEN_PORT)
