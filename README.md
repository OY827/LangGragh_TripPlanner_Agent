# 多智能体智能旅行规划系统

基于 **LangGraph / LangChain** 的中文旅行规划助手。输入城市、日期、天数、交通和偏好后，系统并行检索景点、天气和酒店，用高德回填真实路程，生成按天拆开的 JSON 行程；之后可以多轮对话修改，并保留版本。

## 规划怎么走

Web 主路径是 LangGraph，不是单独的 ReAct 流水线。

```
init → memory → 拼景点查询
                 ├─ 景点子图（知识库检索 ∥ 高德 POI）→ 合并 → 反思 → 整理成推荐文
                 ├─ 天气子图（高德天气）
                 └─ 酒店子图（ReAct + 酒店搜索工具）
                         ↓ 三路到齐
                       高德路径规划 → 行程规划 → 反思
                         ↓ 通过
                       写入长期记忆 → Redis 保存 v1 → 返回
```

景点子图里，Hybrid RAG 与高德 POI 同时跑。结果太少时反思节点会重试。规划结果天数不够、或某一天景点过少时，主图会回到检索或重新生成。

图状态用 LangGraph Checkpointer 保存：配置了 Redis 时用 `RedisSaver`，否则用进程内 `MemorySaver`。

## Hybrid RAG

景点知识库存本地 **Chroma**（集合 `travel_knowledge`）。当前仓库约 **27 个城市、1700 个景点**，一条景点一条文档。`VECTOR_BACKEND=qdrant` 时可改用 Qdrant。

检索顺序：

1. LLM 把用户查询改写成几个同义短查询
2. 向量检索与 BM25 按标题加权混合，并用城市、分类等 metadata 过滤
3. BGE Reranker 精排，再按亲子、历史文化等意图加减分

向量化的是景点正文，不是 metadata。默认嵌入模型是 DashScope `text-embedding-v1`。攻略文本若已灌入，放在另一个集合 `travel_guides`，只在生成行程时作片区串线参考，不能替代景点候选。

灌库：

```bash
python rebuild_kb.py
```

## 记忆

三块分开存。

| 用途 | 存储 | 说明 |
| --- | --- | --- |
| 用户长期记忆 | 本地 Chroma `user_memory` | 行程、已访问景点、偏好。规划前按用户和城市避开去过的景点，规划后写入 |
| 计划版本 | Redis | 会话与 v1、v2… 全文，支持增量修改、对比、回退。Redis 不可用时回退本地 JSON |
| 对话上下文 | 进程内滑动窗口 | LangChain `trim_messages` 保留最近若干轮，再压缩后送给意图识别 |

对话修改走 `DialogAgent`：识别意图后做增量修改或整计划重生成，差异写入新版本。Web 上对应 `/api/dialog/session` 与 `/api/dialog/chat`。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 编排 | LangGraph（并行子图、反思重试、Checkpointer） |
| Agent / 工具 | LangChain；酒店子图为 ReAct + 工具调用 |
| LLM | 通义千问（DashScope，OpenAI 兼容接口） |
| 向量库 | Chroma（默认）；Qdrant 可选 |
| 检索 | DashScope Embedding、BM25、BAAI/bge-reranker-base |
| 地图 | 高德 REST；可选 Streamable HTTP MCP |
| 酒店 | RollingGo 酒店 MCP，高德 POI 兜底 |
| 服务 | FastAPI + 静态页面 |
| 观测 | LangSmith（可选） |

## 快速开始

```bash
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，至少填写：

```
DASHSCOPE_API_KEY=your_key
DASHSCOPE_MODEL=qwen-turbo
AMAP_MAPS_API_KEY=your_amap_key
```

长期记忆和景点检索还需要先把知识库写入本地 Chroma（见上面的 `rebuild_kb.py`）。Redis 用于计划版本，不启动时版本管理会落到 `app/data/` 下的 JSON。

启动：

```bash
python web/main.py
```

浏览器打开 `http://127.0.0.1:8000`。

## 主要 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | Web 页面 |
| GET | `/health` | 健康检查 |
| POST | `/api/trip/plan` | 生成多日行程 |
| POST | `/api/dialog/session` | 为已有计划创建修改会话 |
| POST | `/api/dialog/chat` | 多轮修改 |
| GET | `/api/memory/{user_id}/{city}` | 同城历史记忆 |
| GET | `/api/memory/user/{user_id}` | 该用户的历史计划 |
| POST | `/api/route` | 高德路径规划 |
| GET | `/api/weather/{city}` | 天气 |
| GET | `/api/monitoring/langsmith` | LangSmith 追踪状态 |

## 目录

```
app/agents/          LangGraph 图、节点、对话 Agent、版本管理
app/rag/             Chroma、Hybrid RAG、嵌入、检索
app/memory/          长期记忆、滑动窗口
app/prompts/         按角色拆分的 markdown 提示词
app/services/        高德与酒店 MCP
app/knowledge_base/  景点 JSON，灌库前的源数据
web/                 FastAPI 与页面
scripts/             灌库与评测
tests/               测试
```

## 注意

不要把 `.env` 提交到仓库。Chroma 数据在 `app/data/chroma/`，已加入 `.gitignore`。首次检索或精排若本机没有 BGE Reranker，会从 Hugging Face 下载 `BAAI/bge-reranker-base`。
