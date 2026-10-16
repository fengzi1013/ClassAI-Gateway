from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import httpx
import json
import logging
import time
import csv
from io import StringIO
from collections import defaultdict, deque

# ========= 【部署前必须修改配置】==========
DEEPSEEK_KEY = "sk-替换成你的Deepseek API密钥"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8000
TIMEOUT = 120
# 限流：单学生IP每分钟最大提问次数
RATE_LIMIT_PER_MIN = 6
# IP白名单，不受限流+拉黑限制（填写教师机内网IP）
WHITE_LIST = ["127.0.0.1", "6.163.159.195"]
# 教师后台登录密码
ADMIN_PASSWORD = "teacher123"
# ======================================

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="机房内网AI助教网关")

# IP限流存储：ip -> 请求时间队列
ip_req_records = defaultdict(deque)
# 学生访问统计：ip -> {count, last_time, last_msg, prompt_tokens, completion_tokens, total_tokens}
student_stats = defaultdict(lambda: {
    "total":0,
    "last_time":0,
    "last_msg":"",
    "prompt_tokens":0,
    "completion_tokens":0,
    "total_tokens":0
})
# IP黑名单：拉黑的学生IP，直接拒绝访问
black_ip_set = set()


def check_rate_limit(client_ip: str) -> tuple[bool, str]:
    """
    访问权限校验：
    1. 黑名单IP直接拒绝
    2. 白名单直接放行
    3. 其余IP执行访问频率限制
    """
    if client_ip in black_ip_set:
        return False, "该IP已被管理员限制访问，请联系老师"
    if client_ip in WHITE_LIST:
        return True, ""
    now = time.time()
    queue = ip_req_records[client_ip]
    # 清理超过60秒的旧请求记录
    while queue and queue[0] <= now - 60:
        queue.popleft()
    if len(queue) >= RATE_LIMIT_PER_MIN:
        return False, f"请求限流：每台电脑每分钟最多{RATE_LIMIT_PER_MIN}次提问，请稍后重试！"
    queue.append(now)
    return True, ""


async def stream_generator(payload: dict, client_ip:str):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json"
    }
    # 记录学生提问预览文本
    user_msg = ""
    try:
        for msg in payload["messages"]:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    txt_parts = [c["text"] for c in content if c["type"]=="text"]
                    user_msg = " ".join(txt_parts)
                else:
                    user_msg = str(content)
                break
        student_stats[client_ip]["total"] += 1
        student_stats[client_ip]["last_time"] = time.time()
        student_stats[client_ip]["last_msg"] = user_msg[:120]
    except:
        pass

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream("POST", DEEPSEEK_URL, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    err_text = await resp.aread()
                    yield f"data: {json.dumps({'error':f'上游API错误{resp.status_code}:{err_text.decode()}'})}\n\n"
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
                            # 捕获usage token统计（只有流式返回最后一条带usage）
                            if "usage" in j:
                                usage = j["usage"]
                                pt = usage.get("prompt_tokens",0)
                                ct = usage.get("completion_tokens",0)
                                tt = usage.get("total_tokens",0)
                                student_stats[client_ip]["prompt_tokens"] += pt
                                student_stats[client_ip]["completion_tokens"] += ct
                                student_stats[client_ip]["total_tokens"] += tt
                        except Exception:
                            continue
    except Exception as e:
        yield f"data: {json.dumps({'error':f'网关异常：{str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(request: Request):
    client_ip = request.client.host
    allow, msg = check_rate_limit(client_ip)
    if not allow:
        async def error_stream():
            yield f"data: {json.dumps({'error':msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    body = await request.json()
    body["stream"] = True
    return StreamingResponse(
        stream_generator(body, client_ip),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# ========== 教师后台相关接口 ==========
@app.get("/admin")
async def admin_page():
    return FileResponse("admin.html")

# 获取全部学生统计数据
@app.get("/api/admin/stats")
async def get_stats(pwd:str):
    if pwd != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="密码错误，拒绝访问")
    result = []
    now = time.time()
    for ip, data in student_stats.items():
        diff_sec = int(now - data["last_time"])
        if data["last_time"] == 0:
            last_str = "暂无记录"
        else:
            last_str = f"{diff_sec}秒前" if diff_sec<60 else f"{diff_sec//60}分{diff_sec%60}秒前"
        queue = ip_req_records[ip]
        curr_in_min = len(queue)
        is_limited = curr_in_min >= RATE_LIMIT_PER_MIN
        is_black = ip in black_ip_set
        result.append({
            "ip":ip,
            "total":data["total"],
            "last_active": last_str,
            "last_msg": data["last_msg"],
            "current_min_req": curr_in_min,
            "is_limited": is_limited,
            "is_black": is_black,
            "prompt_tokens": data["prompt_tokens"],
            "completion_tokens": data["completion_tokens"],
            "total_tokens": data["total_tokens"]
        })
    result.sort(key=lambda x: x["total"], reverse=True)
    global_total_token = sum(r["total_tokens"] for r in result)
    return {"list":result, "max_per_min":RATE_LIMIT_PER_MIN, "global_total_token":global_total_token}

# 接口：导出CSV报表
@app.get("/api/admin/export-csv")
async def export_csv(pwd:str):
    if pwd != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="密码错误")
    output = StringIO()
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
            last_str = f"{diff_sec}秒前" if diff_sec<60 else f"{diff_sec//60}分{diff_sec%60}秒前"
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
    return PlainTextResponse(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="ai_student_report.csv"'}
    )

# 接口：重置指定IP的限流计数器
@app.post("/api/admin/reset-limit")
async def reset_limit(pwd:str, ip:str):
    if pwd != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="密码错误")
    if ip in WHITE_LIST:
        return {"ok":False, "msg":"白名单教师IP无需重置"}
    ip_req_records[ip].clear()
    return {"ok":True, "msg":f"已重置IP {ip} 的限流计数"}

# 接口：拉黑 / 解除拉黑IP
@app.post("/api/admin/toggle-black")
async def toggle_black(pwd:str, ip:str):
    if pwd != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="密码错误")
    if ip in WHITE_LIST:
        return {"ok":False, "msg":"不能拉黑白名单教师IP"}
    if ip in black_ip_set:
        black_ip_set.remove(ip)
        return {"ok":True, "msg":f"已解除IP {ip} 的访问限制"}
    else:
        black_ip_set.add(ip)
        return {"ok":True, "msg":f"已拉黑IP {ip}，该学生无法继续提问"}

# ⚠️重要！静态网页挂载必须放在所有API路由之后！
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=LISTEN_HOST, port=LISTEN_PORT)
