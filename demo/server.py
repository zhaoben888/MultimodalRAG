# -*- coding: utf-8 -*-
import os
import re
import uuid
import shutil
import asyncio
import subprocess
import httpx
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any, Optional


app = FastAPI(title="MinerU Web RAG Assistant")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 基础目录配置
DEMO_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = DEMO_DIR / "uploaded_pdfs"
PARSE_OUTPUT_ROOT = DEMO_DIR / "parsed_outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PARSE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app.mount("/parsed_outputs", StaticFiles(directory=str(PARSE_OUTPUT_ROOT)), name="parsed_outputs")


# 支持的文档后缀
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg"}

# 内存向量数据库结构
# doc_id -> { "filename": str, "chunks": [{"text": str, "vector": list, "metadata": dict}] }
VECTOR_DB: Dict[str, Dict[str, Any]] = {}
DB_FILE = DEMO_DIR / "vector_db.json"

API_KEYS_FILE = DEMO_DIR / "api_keys.json"
API_KEYS = {"bailian": "", "deepseek": ""}

def load_api_keys():
    global API_KEYS
    if API_KEYS_FILE.exists():
        try:
            import json as _j
            API_KEYS = _j.loads(API_KEYS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

def save_api_keys(data: dict):
    global API_KEYS
    API_KEYS = data
    API_KEYS_FILE.write_text(
        __import__("json").dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

load_api_keys()

def _relink_images(doc_id: str, force: bool = False) -> bool:
    """
    将 doc_id 嵌套目录下散落的 images/ 合并到 parsed_outputs/{doc_id}/images/。
    判定依据：当顶层 images 不存在/为空，或顶层 images 中的文件数小于 chunk 引用数时触发。
    返回 True 表示发生了实际合并。
    """
    doc_dir = PARSE_OUTPUT_ROOT / doc_id
    if not doc_dir.exists() or not doc_dir.is_dir():
        return False
    target_img_dir = doc_dir / "images"
    target_img_dir.mkdir(parents=True, exist_ok=True)

    def _count_files(d: Path) -> int:
        try:
            return sum(1 for _ in d.iterdir())
        except Exception:
            return 0

    def _chunk_image_refs() -> set:
        refs = set()
        doc = VECTOR_DB.get(doc_id)
        if not doc:
            return refs
        for c in doc.get("chunks", []):
            for m in re.finditer(r'images/([0-9a-f]+\.jpg)', c.get("text", "")):
                refs.add(m.group(1))
        return refs

    cur_count = _count_files(target_img_dir)
    if force:
        need_fix = True
    elif cur_count == 0:
        need_fix = True
    else:
        refs = _chunk_image_refs()
        if refs:
            cur_names = {p.name for p in target_img_dir.iterdir() if p.is_file()}
            need_fix = bool(refs - cur_names)
        else:
            need_fix = False

    if not need_fix:
        return False

    print(f"[relink] fixing images for {doc_id}")
    nested_images_dirs = [d for d in doc_dir.glob("**/images") if d.is_dir() and d != target_img_dir]
    fixed = False
    for src_dir in nested_images_dirs:
        for f in src_dir.iterdir():
            if f.is_file():
                dst = target_img_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    fixed = True
    return fixed


def save_vector_db():
    try:
        import json
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(VECTOR_DB, f, ensure_ascii=False, indent=2)
        print("Vector database saved successfully.")
    except Exception as e:
        print(f"Failed to save vector DB: {e}")

def load_vector_db():
    global VECTOR_DB
    if DB_FILE.exists():
        try:
            import json
            with open(DB_FILE, "r", encoding="utf-8") as f:
                VECTOR_DB = json.load(f)
            print(f"Loaded {len(VECTOR_DB)} documents from persistent vector DB")

            # 启动时自检：用统一 _relink_images 修复所有文档的嵌套图片目录
            try:
                for doc_dir in PARSE_OUTPUT_ROOT.iterdir():
                    if doc_dir.is_dir() and doc_dir.name in VECTOR_DB:
                        _relink_images(doc_dir.name)
            except Exception as fix_err:
                print(f"Failed to auto-fix nested image paths: {fix_err}")

        except Exception as e:
            print(f"Failed to load vector DB: {e}")

def kill_zombie_mineru_api():
    if os.name == 'nt':
        try:
            # 精确匹配 mineru.cli 以强杀提取子进程，避免误杀带有 mineru_env 虚拟环境路径的主进程
            cmd = 'wmic process where "name=\'python.exe\' and CommandLine like \'%mineru.cli%\'" get ProcessId'
            out = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
            pids = [int(p) for p in re.findall(r'\d+', out)]
            my_pid = os.getpid()
            for pid in pids:
                if pid != my_pid:
                    print(f"Killing zombie mineru-api process with PID: {pid}")
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Failed to kill zombie processes: {e}")

# 初始化加载向量库并清理残留进程
load_vector_db()

# 启动时无条件兼容修复旧版嵌套图片路径
try:
    print("=== STARTING IMAGE PATH AUTO-FIX CHECK ===")
    if PARSE_OUTPUT_ROOT.exists():
        for doc_dir in PARSE_OUTPUT_ROOT.iterdir():
            if doc_dir.is_dir():
                _relink_images(doc_dir.name)
            else:
                pass
except Exception as e:
    print(f"Error during image path auto-fix: {e}")

kill_zombie_mineru_api()

# 本地目录挂载解析任务状态
MOUNT_STATUS = {
    "status": "idle",       # "idle", "running", "completed", "error"
    "total": 0,
    "completed": 0,
    "current_file": "",
    "error_msg": ""
}

class ChatRequest(BaseModel):
    doc_id: str
    question: str
    img_corrections: Optional[dict] = None
    product_images: Optional[dict] = None

class MountRequest(BaseModel):
    folder_path: str

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def parse_markdown_to_chunks(md_content: str, max_chunk_size: int = 800) -> List[Dict[str, Any]]:
    lines = md_content.split('\n')
    chunks = []
    current_header = "默认章节"
    current_parent_header = "通用章节"
    current_chunk = []
    current_size = 0
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            
            if current_chunk:
                # 构建复合header：子标题前面拼接产品名，提升检索召回
                if current_parent_header.startswith("GB") and not current_header.startswith("GB"):
                    display_header = f"{current_parent_header} - {current_header}"
                else:
                    display_header = current_header
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "metadata": {"header": display_header, "parent_header": current_parent_header}
                })
                current_chunk = []
                current_size = 0
            
            # 遇到GB开头的标题就更新父级（无论层级）, 覆盖 GB系列、GB4911、GB2001 等
            if header_text.startswith("GB"):
                current_parent_header = header_text
            current_header = header_text
            current_chunk.append(line)
            current_size += len(line)
        else:
            current_chunk.append(line)
            current_size += len(line)
            if current_size >= max_chunk_size:
                if current_parent_header.startswith("GB") and not current_header.startswith("GB"):
                    display_header = f"{current_parent_header} - {current_header}"
                else:
                    display_header = current_header
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "metadata": {"header": display_header, "parent_header": current_parent_header}
                })
                current_chunk = []
                current_size = 0
    if current_chunk:
        if current_parent_header.startswith("GB") and not current_header.startswith("GB"):
            display_header = f"{current_parent_header} - {current_header}"
        else:
            display_header = current_header
        chunks.append({
            "text": "\n".join(current_chunk),
            "metadata": {"header": display_header, "parent_header": current_parent_header}
        })
    return [c for c in chunks if len(c["text"].strip()) > 10]

async def get_bailian_embedding(texts: List[str], api_key: str) -> List[List[float]]:
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "text-embedding-v3",
        "input": {
            "texts": texts
        },
        "parameters": {
            "text_type": "document"
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Bailian Embedding API error: {response.text}")
            res_json = response.json()
            embeddings = [item["embedding"] for item in res_json["output"]["embeddings"]]
            return embeddings
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to query Bailian Embedding: {str(e)}")

async def get_bailian_query_embedding(text: str, api_key: str) -> List[float]:
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "text-embedding-v3",
        "input": {
            "texts": [text]
        },
        "parameters": {
            "text_type": "query"
        }
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Bailian Query Embedding API error: {response.text}")
        res_json = response.json()
        return res_json["output"]["embeddings"][0]["embedding"]

# 单文档通用解析建库流程（底层封装）
async def parse_and_index_document(file_path: Path, filename: str, bailian_key: str) -> Dict[str, Any]:
    import tempfile
    import ctypes
    from ctypes import wintypes
    doc_id = str(uuid.uuid4())
    
    # 获取 Windows 短路径格式，规避中文路径传参乱码问题
    def get_short_path(long_path_str: str) -> str:
        try:
            _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
            _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            _GetShortPathNameW.restype = wintypes.DWORD
            output_buf_size = 256
            while True:
                output_buf = ctypes.create_unicode_buffer(output_buf_size)
                needed = _GetShortPathNameW(long_path_str, output_buf, output_buf_size)
                if needed == 0:
                    return long_path_str
                if needed < output_buf_size:
                    return output_buf.value
                output_buf_size = needed
        except Exception:
            return long_path_str

    temp_dir_root = Path(tempfile.gettempdir())
    temp_input_path = temp_dir_root / f"mineru_in_{doc_id}{file_path.suffix}"
    temp_out_dir = temp_dir_root / f"mineru_out_{doc_id}"
    temp_out_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(file_path, temp_input_path)
    
    import sys
    python_exe_str = sys.executable
    if os.name == 'nt':
        python_exe_str = get_short_path(python_exe_str)
        
    cmd = [
        python_exe_str,
        "-m", "mineru.cli.client",
        "-p", str(temp_input_path),
        "-o", str(temp_out_dir),
        "-b", "pipeline"
    ]
    
    sub_env = os.environ.copy()
    sys_path = "C:\\Windows\\system32;C:\\Windows;C:\\Windows\\System32\\Wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0"
    if "PATH" in sub_env:
        sub_env["PATH"] = sys_path + ";" + sub_env["PATH"]
    else:
        sub_env["PATH"] = sys_path

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=sub_env
        )
        try:
            # 设置 300 秒执行超时
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300.0)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception:
                pass
            raise Exception("MinerU 解析文档超时（超过300秒）")

        if process.returncode != 0:
            err_msg = stderr.decode('utf-8', errors='ignore').strip()
            if not err_msg:
                err_msg = stdout.decode('utf-8', errors='ignore').strip()
            if not err_msg:
                err_msg = f"进程意外闪退 (退出码: {process.returncode})，可能由于系统组件冲突"
            raise Exception(err_msg)
    except Exception as e:
        shutil.rmtree(temp_out_dir, ignore_errors=True)
        temp_input_path.unlink(missing_ok=True)
        try:
            log_file = DEMO_DIR / "debug_error.log"
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"=== Error for {filename} ===\n{str(e)}\n\n")
        except Exception:
            pass
        raise Exception(f"MinerU 提取排版失败: {str(e)}")
    finally:
        # 强制清理 Windows 下可能残留的子进程，释放 GPU 显存
        kill_zombie_mineru_api()
        
    md_files = list(temp_out_dir.glob("**/*.md"))
    if not md_files:
        shutil.rmtree(temp_out_dir, ignore_errors=True)
        temp_input_path.unlink(missing_ok=True)
        raise Exception("未能成功生成 Markdown 文档成果。")

    # 优选 md：MinerU hybrid-engine 默认产物的图片与同名 .md 同位于 {stem}/auto/ 目录下，
    # 优先选择 auto/ 路径下的 md 以确保其相邻 Jun 25-2026 年底最新的图片集合
    def _md_priority(p: Path) -> tuple:
        parts = p.parts
        is_auto = "auto" in parts
        depth = len(parts)
        return (0 if is_auto else 1, depth, str(p))

    md_file_path = sorted(md_files, key=_md_priority)[0]

    # 保存提取出的产品图片到永久存储区
    doc_parse_dir = PARSE_OUTPUT_ROOT / doc_id
    doc_parse_dir.mkdir(parents=True, exist_ok=True)
    target_img_dir = doc_parse_dir / "images"
    target_img_dir.mkdir(parents=True, exist_ok=True)

    all_img_dirs = [p for p in temp_out_dir.glob("**/images") if p.is_dir()]
    copied_count = 0
    for img_src_dir in all_img_dirs:
        try:
            for f in img_src_dir.iterdir():
                if f.is_file():
                    dst = target_img_dir / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
                        copied_count += 1
        except Exception as img_err:
            print(f"Failed to copy images from {img_src_dir}: {img_err}")

    # 读取并改写md引用
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    # 改写HTML img标签
    md_content = re.sub(
        r'(<img\s[^>]*src=")(?:images/)?([^"]+)(")',
        rf'\1/parsed_outputs/{doc_id}/images/\2\3',
        md_content
    )
    # 改写markdown图片语法
    md_content = re.sub(
        r'!\[(.*?)\]\((?:images/)?([^)]+)\)',
        rf'![\1](/parsed_outputs/{doc_id}/images/\2)',
        md_content
    )
    _refs = set(re.findall(r'images/([0-9a-f]+\.jpg)', md_content))
    _actual = {p.name for p in target_img_dir.iterdir()} if target_img_dir.exists() else set()
    _missing = _refs - _actual
    if _missing:
        try:
            with open(DEMO_DIR / "debug_error.log", "a", encoding="utf-8") as lf:
                lf.write(f"=== Image miss for {filename} ({doc_id}) ===\n"
                         f"refs={len(_refs)} actual={len(_actual)} missing={len(_missing)}\n"
                         f"missing_sample={list(_missing)[:5]}\n\n")
        except Exception:
            pass
    else:
        print(f"Image check OK for {filename}: refs={len(_refs)} copied={copied_count}")
        
    shutil.rmtree(temp_out_dir, ignore_errors=True)
    temp_input_path.unlink(missing_ok=True)
        
    chunks = parse_markdown_to_chunks(md_content)
    if not chunks:
        raise Exception("文档解析后没有发现有效的内容段落。")
        
    for i, c in enumerate(chunks):
        clean_filename = Path(filename).stem
        parent_h = c["metadata"].get("parent_header", "通用章节")
        # 强行拼入父级标题（产品名字），保证切片被切碎后依然能够保留产品归属上下文
        c["text"] = f"[背景文档: {clean_filename} | 产品/主题: {parent_h} | 章节: {c['metadata']['header']}]\n{c['text']}"
        c["chunk_idx"] = i
        
    chunk_texts = [c["text"] for c in chunks]
    embeddings = []
    batch_size = 10
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]
        batch_embeddings = await get_bailian_embedding(batch, bailian_key)
        embeddings.extend(batch_embeddings)
        
    db_entry = {
        "filename": filename,
        "chunks": []
    }
    for chunk, embedding in zip(chunks, embeddings):
        db_entry["chunks"].append({
            "text": chunk["text"],
            "vector": embedding,
            "metadata": chunk["metadata"],
            "chunk_idx": chunk.get("chunk_idx", 0)
        })
        
    VECTOR_DB[doc_id] = db_entry
    save_vector_db()  # 保存持久化向量数据库
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks)
    }

@app.post("/api/upload")
async def upload_batch(
    files: List[UploadFile] = File(...),
    x_bailian_key: Optional[str] = Header(None),
):
    if not x_bailian_key or x_bailian_key == "undefined" or len(x_bailian_key.strip()) < 5:
        raise HTTPException(status_code=400, detail="需要提供有效的百炼 API Key")
        
    results = []
    for file in files:
        # 只处理支持的类型
        suffix = Path(file.filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
            
        doc_uuid = str(uuid.uuid4())
        temp_path = UPLOAD_DIR / f"{doc_uuid}{suffix}"
        
        # 1. 保存临时上传文件
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            # 2. 解析建库
            res = await parse_and_index_document(temp_path, file.filename, x_bailian_key)
            results.append(res)
        except Exception as e:
            # 记录失败文件并继续
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
        finally:
            # 清理临时上传文件
            temp_path.unlink(missing_ok=True)
            
    return results

async def bg_mount_task(folder_path: str, bailian_key: str):
    global MOUNT_STATUS
    MOUNT_STATUS["status"] = "running"
    MOUNT_STATUS["error_msg"] = ""
    
    try:
        path = Path(folder_path).resolve()
        # 扫描文件
        files_to_parse = []
        for item in path.iterdir():
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES:
                # 判断是否已经存在于内存向量库（通过文件名去重）
                already_loaded = any(v["filename"] == item.name for v in VECTOR_DB.values())
                if not already_loaded:
                    files_to_parse.append(item)
                    
        MOUNT_STATUS["total"] = len(files_to_parse)
        MOUNT_STATUS["completed"] = 0
        
        if len(files_to_parse) == 0:
            MOUNT_STATUS["status"] = "completed"
            return
            
        for file in files_to_parse:
            MOUNT_STATUS["current_file"] = file.name
            try:
                await parse_and_index_document(file, file.name, bailian_key)
            except Exception as e:
                # 记录警告，不中断整体挂载
                print(f"Mount parse failed for {file.name}: {str(e)}")
            MOUNT_STATUS["completed"] += 1
            
        MOUNT_STATUS["status"] = "completed"
        MOUNT_STATUS["current_file"] = ""
        
    except Exception as e:
        MOUNT_STATUS["status"] = "error"
        MOUNT_STATUS["error_msg"] = str(e)

@app.post("/api/mount")
async def mount_local_folder(
    request: MountRequest,
    background_tasks: BackgroundTasks,
    x_bailian_key: Optional[str] = Header(None),
):
    global MOUNT_STATUS
    if not x_bailian_key or x_bailian_key == "undefined" or len(x_bailian_key.strip()) < 5:
        raise HTTPException(status_code=400, detail="请配置百炼 API Key Header。")
        
    if MOUNT_STATUS["status"] == "running":
        raise HTTPException(status_code=400, detail="当前已有其他文件夹扫描挂载任务正在后台运行，请勿重复发起。")
        
    path = Path(request.folder_path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="指定的本地目录不存在或非有效文件夹，请确认路径正确性。")
        
    # 添加至后台处理，立即返回响应以防止 HTTP 连接超时
    background_tasks.add_task(bg_mount_task, str(path), x_bailian_key)
    return {"message": "本地挂载扫描任务已在后台启动。"}

@app.get("/api/mount/status")
async def get_mount_status():
    return MOUNT_STATUS

@app.get("/api/documents")
async def get_documents():
    result = []
    for k, v in VECTOR_DB.items():
        # 自检：chunk 引用的图片是否都存在
        refs = set()
        for c in v.get("chunks", []):
            for m in re.finditer(r'images/([0-9a-f]+\.jpg)', c.get("text", "")):
                refs.add(m.group(1))
        img_dir = PARSE_OUTPUT_ROOT / k / "images"
        actual = {p.name for p in img_dir.iterdir()} if img_dir.exists() and img_dir.is_dir() else set()
        images_ok = (not refs) or bool(refs <= actual)
        result.append({
            "doc_id": k,
            "filename": v["filename"],
            "chunks": len(v["chunks"]),
            "images_ok": images_ok,
            "images_refs": len(refs),
            "images_present": len(actual),
        })
    return result

@app.get("/api/documents/{doc_id}/images")
async def get_document_images(doc_id: str):
    # 全局知识库无需展示图片
    if doc_id == "all" or doc_id == "global":
        return []
        
    if doc_id not in VECTOR_DB:
        raise HTTPException(status_code=404, detail="未找到指定的文档。")
        
    doc_dir = PARSE_OUTPUT_ROOT / doc_id / "images"
    if not doc_dir.exists() or not doc_dir.is_dir():
        return []
        
    img_files = []
    for item in doc_dir.iterdir():
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}:
            img_files.append(item.name)
            
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        
    img_files.sort(key=natural_sort_key)
    return [f"/parsed_outputs/{doc_id}/images/{name}" for name in img_files]


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in VECTOR_DB:
        raise HTTPException(status_code=404, detail="未找到指定的文档。")
    filename = VECTOR_DB[doc_id]["filename"]
    del VECTOR_DB[doc_id]
    save_vector_db()
    doc_dir = PARSE_OUTPUT_ROOT / doc_id
    if doc_dir.exists() and doc_dir.is_dir():
        shutil.rmtree(doc_dir, ignore_errors=True)
    return {"deleted": doc_id, "filename": filename}


@app.get("/api/keys")
async def get_api_keys():
    return {
        "bailian": API_KEYS.get("bailian", ""),
        "deepseek": API_KEYS.get("deepseek", "")
    }

@app.post("/api/keys")
async def post_api_keys(request: dict):
    bailian = request.get("bailian", "")
    deepseek = request.get("deepseek", "")
    save_api_keys({"bailian": bailian, "deepseek": deepseek})
    return {"ok": True}

@app.get("/api/debug/log")
async def get_debug_log():
    log_file = DEMO_DIR / "debug_error.log"
    if not log_file.exists():
        return {"log": "暂无日志信息。说明目前尚未产生解析或调用错误。"}
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        # 返回最近 20000 个字符
        return {"log": content[-20000:]}
    except Exception as e:
        return {"log": f"读取日志失败: {str(e)}"}

@app.post("/api/debug/clear_log")
async def clear_debug_log():
    log_file = DEMO_DIR / "debug_error.log"
    if log_file.exists():
        try:
            log_file.unlink()
            return {"message": "日志已清空"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"无法清空日志: {str(e)}")
    return {"message": "日志文件不存在，无需清空"}

@app.get("/api/debug/vector_db_info")
async def get_vector_db_info():
    info = []
    for doc_id, doc_data in VECTOR_DB.items():
        total_chars = sum(len(c["text"]) for c in doc_data["chunks"])
        info.append({
            "doc_id": doc_id,
            "filename": doc_data["filename"],
            "chunk_count": len(doc_data["chunks"]),
            "total_chars": total_chars
        })
    return info

@app.get("/api/debug/chunks/{doc_filename}")
async def get_doc_chunks(doc_filename: str):
    chunks = []
    for doc_id, doc_data in VECTOR_DB.items():
        if doc_data["filename"] == doc_filename:
            for idx, c in enumerate(doc_data["chunks"]):
                chunks.append({
                    "chunk_index": idx + 1,
                    "header": c["metadata"]["header"],
                    "text": c["text"]
                })
    return {"filename": doc_filename, "total_chunks": len(chunks), "chunks": chunks}

@app.post("/api/chat")
async def chat_rag(
    request: ChatRequest,
    x_bailian_key: Optional[str] = Header(None),
    x_deepseek_key: Optional[str] = Header(None),
):
    if not x_bailian_key or x_bailian_key == "undefined" or not x_deepseek_key or x_deepseek_key == "undefined":
        raise HTTPException(status_code=400, detail="需要提供合法的百炼 API Key 及 DeepSeek API Key Headers")
        
    doc_id = request.doc_id
    query_vector = await get_bailian_query_embedding(request.question, x_bailian_key)
    query_vector_np = np.array(query_vector)
    
    # 提取查询中的关键词以计算关键字加成（Hybrid Score）
    q_lower = request.question.lower()
    eng_num_keywords = re.findall(r'[a-zA-Z0-9]+', q_lower)
    han_keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', q_lower)
    all_keywords = list(set(eng_num_keywords + han_keywords))
    
    scored_chunks = []
    
    if doc_id == "all" or doc_id == "global":
        for d_id, doc_data in VECTOR_DB.items():
            for chunk in doc_data["chunks"]:
                chunk_vector_np = np.array(chunk["vector"])
                sim = cosine_similarity(query_vector_np, chunk_vector_np)
                
                # 计算关键字加成权重
                boost = 0.0
                chunk_text_lower = chunk["text"].lower()
                for kw in all_keywords:
                    if kw in chunk_text_lower:
                        if re.match(r'^[a-zA-Z0-9]+$', kw) and len(kw) >= 3:
                            boost += 0.20
                        else:
                            boost += 0.05
                
                final_score = sim + min(boost, 0.4)
                chunk_copy = chunk.copy()
                chunk_copy["filename"] = doc_data["filename"]
                chunk_copy["doc_id"] = d_id
                scored_chunks.append((final_score, chunk_copy))
    else:
        # 单文档检索
        if doc_id not in VECTOR_DB:
            raise HTTPException(status_code=404, detail="未找到指定的文档向量库，请重新上传文件。")
        doc_data = VECTOR_DB[doc_id]
        for chunk in doc_data["chunks"]:
            chunk_vector_np = np.array(chunk["vector"])
            sim = cosine_similarity(query_vector_np, chunk_vector_np)
            
            # 计算关键字加成权重
            boost = 0.0
            chunk_text_lower = chunk["text"].lower()
            for kw in all_keywords:
                if kw in chunk_text_lower:
                    if re.match(r'^[a-zA-Z0-9]+$', kw) and len(kw) >= 3:
                        boost += 0.15
                    else:
                        boost += 0.05
                        
            final_score = sim + min(boost, 0.4)
            chunk_copy = chunk.copy()
            chunk_copy["filename"] = doc_data["filename"]
            chunk_copy["doc_id"] = doc_id
            scored_chunks.append((final_score, chunk_copy))
        
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    # 扩大召回上限，防止关键参数被漏掉，且DeepSeek大模型具有强大的长上下文处理能力
    top_k_count = 15 if (doc_id == "all" or doc_id == "global") else 12
    top_chunks = [item[1] for item in scored_chunks[:top_k_count]]
    
    for chunk in top_chunks:
        c_doc_id = chunk.get("doc_id", "")
        if c_doc_id:
            # 准时制自动兼容修复（JIT Fix）：调用统一 _relink_images，按 chunk 引用集合判断是否需要修复
            try:
                _relink_images(c_doc_id)
            except Exception as jit_err:
                print(f"JIT Fix failed for {c_doc_id}: {jit_err}")
                
            # 使用负向先行断言规避对已是绝对路径的图片进行二次替换
            chunk["text"] = re.sub(
                r'!\[(.*?)\]\s*\((?!/parsed_outputs)(?:images/)?(.*?)\)',
                rf'![\1](/parsed_outputs/{c_doc_id}/images/\2)',
                chunk["text"]
            )
            # 同时处理HTML img标签，同样规避已转换的绝对路径
            chunk["text"] = re.sub(
                r'(<img\s[^>]*src=")(?!/parsed_outputs)(?:images/)?([^"]+)(")',
                rf'\1/parsed_outputs/{c_doc_id}/images/\2\3',
                chunk["text"]
            )
            
    context_str = "\n\n".join([
        f"[参考来源文档: {chunk.get('filename', '未知')} | 章节: {chunk['metadata']['header']} | 匹配序号: {idx+1}]:\n{chunk['text']}"
        for idx, chunk in enumerate(top_chunks)
    ])
    
    system_prompt = f"""你是产品知识库问答助手。只能根据下方背景知识回答，不要编造信息。

规则：
1. 数值筛选条件必须严格执行，只输出符合条件的行，不要把不合格的列出来标注
2. 参数罗列用 Markdown 表格
3. 【最重要】回答中涉及图片时，必须逐字复制背景知识里的 /parsed_outputs/... 图片URL，禁止修改文件名（文件名是hash值，不是产品名）
4. 简洁回答，不要冗余铺垫

背景知识：
==================================
{context_str}
==================================
"""


    async def response_stream():
        import json
        sources_payload = [
            {
                "index": idx+1, 
                "header": f"《{chunk.get('filename', '未知')}》- {chunk['metadata']['header']}", 
                "text": chunk["text"][:300] + "..."
            }
            for idx, chunk in enumerate(top_chunks)
        ]
        yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"
        
        deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {x_deepseek_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ],
            "stream": True,
            "temperature": 0.0
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream("POST", deepseek_url, headers=headers, json=body) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        yield f"event: error\ndata: DeepSeek API error: {err_text.decode('utf-8')}\n\n"
                        return
                        
                    async for chunk in response.aiter_lines():
                        if not chunk.strip():
                            continue
                        if chunk.startswith("data: "):
                            data_str = chunk[6:]
                            if data_str == "[DONE]":
                                break
                            data_json = json.loads(data_str)
                            delta = data_json["choices"][0]["delta"]
                            if "content" in delta:
                                content = delta["content"]
                                yield f"data: {json.dumps({'content': content})}\n\n"
            except Exception as e:
                yield f"event: error\ndata: HTTP request exception: {str(e)}\n\n"
                
    return StreamingResponse(response_stream(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = DEMO_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Frontend index.html is loading...</h3>"

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
