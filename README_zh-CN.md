# MultimodalRAG - 企业级多模态图文问答 RAG 智能助手

MultimodalRAG 是一款基于 **MinerU 文档解析引擎** 构建的本地多模态图文检索与问答系统（RAG Agent）。系统集成了高精度文档解析、智能分块元数据增强、语义向量+关键词加权的混合检索算法，并对接 DeepSeek 大模型实现流式的图文对齐问答。

本仓库旨在提供一个开箱即用的本地多模态图文知识库方案，特别针对 PDF 中的复杂表格、插图在 RAG 中的对齐检索与渲染展示进行了深度工程优化。

---

## 🚀 核心功能与技术亮点

*   **📌 多模态图文解析与对齐**：调用底层的 **MinerU** 解析引擎，将 PDF、Docx、PPTX 等多格式文档中的文字、复杂表格（转为 HTML）及插图结构化提取；设计并实现**图片与上下文 JIT（准时制）自动对齐修复机制**，确保大模型生成的回答中能准确内嵌渲染插图。
*   **🔍 Hybrid Score 混合检索算法**：基于余弦相似度（Cosine Similarity）进行语义检索，并在此基础上融合针对中英文与数字关键词（如特定国标、产品序列号）的 **Keyword Boost 加权得分算子**，显著提升了参数文档检索的召回精准度。
*   **📂 本地异步扫描与挂载**：支持直接输入本地磁盘目录，基于 FastAPI `BackgroundTasks` 在后台异步扫描、解析并持久化未被索引的文档，实现非阻塞的本地知识库挂载。
*   **⚡ 企业级 Windows 环境适配**：
    *   通过 Windows 短路径 API (`GetShortPathNameW`) 彻底规避中文路径乱码引发的子进程解析错误。
    *   引入僵尸进程主动强杀逻辑（`kill_zombie_mineru_api`），解决本地显存残留与 GPU 资源占用问题。
*   **🎨 现代暗黑风极客前端**：使用 Vanilla CSS 编写了极具科技感的毛玻璃（Glassmorphism）和极光背景动效的前端会话界面，支持流式 SSE (Server-Sent Events) 文本传输、参考源（Sources）前置卡片渲染。

---

## 🛠️ 系统架构与技术栈

*   **前端**：HTML5 / Vanilla CSS (现代暗黑微渐变动效) / ES6 JavaScript / Markdown 与表格渲染器
*   **后端**：FastAPI / Uvicorn / Asyncio / NumPy / Pydantic / httpx
*   **大模型与向量服务**：
    *   **向量 Embedding**：阿里百炼 API (`text-embedding-v3`)
    *   **推理 LLM**：DeepSeek API (`deepseek-chat`)
*   **底层解析基座**：MinerU 开源文档解析引擎

---

## 📦 快速启动与部署

### 1. 安装核心依赖
系统运行依赖于 Python 环境（推荐 `3.10 ~ 3.12`）。请在您的 Python 虚拟环境中安装解析引擎及 Web 服务依赖：

```bash
# 安装 Web 服务与数值计算依赖
pip install fastapi uvicorn numpy httpx pydantic

# 安装 MinerU 解析引擎（包含 CUDA 加速支持，国内用户推荐使用阿里云镜像源）
pip install uv -i https://mirrors.aliyun.com/pypi/simple
uv pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple
```
> *注：首次运行 MinerU 解析时会自动从 ModelScope/HuggingFace 下载所需的版面分析及 OCR 模型权重。*

### 2. 启动本地服务
在项目根目录（`MultimodalRAG` 子文件夹下）的终端中运行：
```bash
python demo/server.py
```
默认服务将在本地启动，浏览器访问：[http://127.0.0.1:8000](http://127.0.0.1:8000)

### 3. 配置与使用
1. 打开网页后，点击右上角的 **“配置 API Key”**：
   * **Bailian API Key**：配置阿里百炼密钥以支持切片的 Embedding 向量检索。
   * **DeepSeek API Key**：配置 DeepSeek 密钥以运行多模态对话。
2. 上传文档或在“本地文件夹挂载”输入路径进行索引。
3. 即可开始流式图文知识检索与问答。

---

## 🔒 敏感信息与安全提示
请确保在本地运行时配置您的 API 密钥。本仓库已配置了 `.gitignore`，**绝对不要**将包含真实密钥的 `demo/api_keys.json` 提交并推送到公共的 GitHub 仓库。
