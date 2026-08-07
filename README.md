# 多智能体智能旅行规划系统

基于 **LangChain / LangGraph** 的中文智能旅行规划助手：多 Agent 协作生成多日行程，结合 **Hybrid RAG 景点知识库**、**高德地图真实路线**、**对话式计划修改**，并用 **Redis 持久化用户长期旅行记忆**（断连时自动回退本地文件）。

---

## 项目简介

输入城市、日期、天数、交通与偏好后，系统自动完成：

1. **检索与推荐** — 从城市景点知识库（Qdrant + BGE / Rerank）与高德 POI 中筛选候选景点  
2. **天气与住宿** — 查询旅行期间天气，并按住宿类型推荐酒店  
3. **行程编排** — 按游玩时长（`visit_duration`）排满全天，每天建议 3–5 个景点，含餐饮与时间表  
4. **真实路线** — 调用高德路径规划（步行 / 驾车 / 公交地铁）回填相邻景点的距离与耗时  
5. **长期记忆** — 将历史计划写入 Redis，再次规划时避开已游玩景点、支持同城计划对比  
6. **对话修改** — 生成后可多轮对话增删景点、调整预算与天数，并保留计划版本  

Web 端入口为 FastAPI + 静态页面；规划主路径为 **ReAct 多智能体**；另提供 **LangGraph** 编排实现（并行子图 + 反思重试），便于扩展与实验。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 多智能体规划 | 景点 / 天气 / 酒店 / 规划 Agent 分工协作（ReAct） |
| Hybrid RAG | 查询改写 → 向量检索 → 相关性校验 → BGE Rerank → 生成 |
| 高德地图 | POI 搜索、天气、步行/驾车/公交路径规划 |
| Redis 长期记忆 | 按用户持久化历史行程，支持检索、对比与差异化推荐 |
| 对话式改计划 | 意图识别 + 结构化编辑 + 版本管理 |
| 可观测性 | 可选 LangSmith 链路追踪与本地监控指标 |

---

## 长期记忆（Redis）

系统通过 `TravelMemoryManager`（`app/memory/memory_manager.py`）管理用户旅行长期记忆，**默认优先使用 Redis**。

### 存什么

- 历史旅行计划：城市、天数、景点列表、每日行程、偏好、预算、备注、创建时间  
- 按用户 / 城市 / 景点建立索引，便于快速查询「去过哪些地方」「同城历史计划」

### Redis Key 设计（示意）

```
travel:user:{user_id}:plans                    # 用户计划 ID 列表
travel:user:{user_id}:plan:{plan_id}           # 计划详情 JSON
travel:user:{user_id}:city:{city}              # 某城市下的计划 ID
travel:user:{user_id}:city:{city}:attraction:{name}  # 景点 → 计划集合
```

### 规划时如何用

1. **规划前**：读取该用户在目标城市的已访问景点，提示 Agent 避免重复推荐  
2. **规划后**：将新计划写入 Redis（或文件），形成可累积的旅行档案  
3. **对比推荐**：同城多份计划可做差异分析，推荐「尚未去过」的景点  

### 降级策略

- Redis 不可用或未配置时，自动回退到本地 JSON：`app/data/travel_plans/{user_id}/`  
- 业务接口不变，本地开发可不启 Redis 也能跑通全流程  

相关环境变量：`REDIS_HOST`、`REDIS_PORT`、`REDIS_PASSWORD`。

> 说明：对话会话的版本管理当前 Web 路径默认使用本地文件；LangGraph 可选 `RedisSaver` 作图状态 Checkpointer。用户级「去过哪 / 历史计划」这一长期记忆主链路由 Redis（或文件回退）承担。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 编排 | LangChain（ReAct / Tool Calling）、LangGraph（并行子图 + Reflect） |
| LLM | 通义千问（DashScope）等，可切换 |
| 向量库 | Qdrant |
| 嵌入 / 重排 | DashScope Embedding 或 `BAAI/bge-base-zh-v1.5`；`BAAI/bge-reranker-base` |
| 地图 | 高德 Web 服务 API（可选 MCP） |
| 长期记忆 | Redis（主）+ 本地 JSON 文件（回退） |
| 图数据（可选） | Neo4j 景点关系模块（`app/kg`，非 Web 主路径） |
| 服务端 | FastAPI + Uvicorn |
| 前端 | `web/static/index.html` |
| 观测 | LangSmith（可选） |

---

## 系统架构（简图）

```
用户（Web UI）
    │
    ▼
FastAPI (web/main.py)
    │
    ├─ POST /api/trip/plan  ──► MultiAgentTripPlanner
    │                              ├─ Memory（Redis 长期记忆）
    │                              ├─ 景点 Agent（RAG + 高德）
    │                              ├─ 天气 / 酒店 Agent（高德）
    │                              ├─ 路线回填（高德 path）
    │                              └─ 规划 Agent → TripPlan
    │
    ├─ POST /api/dialog/*   ──► DialogAgent（多轮改计划 + 版本）
    ├─ GET  /api/memory/*   ──► 历史计划 / 同城对比
    └─ POST /api/route      ──► 高德实时路径
```

---

## 快速开始

### 1. 环境

推荐 Conda 环境 `py312_env`：

```powershell
& "C:\Users\OZ\anaconda3\envs\py312_env\python.exe" -m pip install -r requirements.txt
```

### 2. 配置 `.env`

复制 `.env.example` 为 `.env`，至少配置：

```env
# LLM
DASHSCOPE_API_KEY=your_key

# 高德
AMAP_MAPS_API_KEY=your_amap_key

# 向量库
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis 长期记忆（可选；不配则自动用本地文件）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# LangSmith（可选）
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

### 3. 启动 Web

```powershell
& "C:\Users\OZ\anaconda3\envs\py312_env\python.exe" web/main.py
```

浏览器打开：`http://127.0.0.1:8000`

---

## 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 旅行规划 Web UI |
| GET | `/health` | 健康检查 |
| POST | `/api/trip/plan` | 生成多日旅行计划 |
| POST | `/api/dialog/session` | 创建对话改计划会话 |
| POST | `/api/dialog/chat` | 多轮修改行程 |
| GET | `/api/memory/{user_id}/{city}` | 同城历史记忆与对比 |
| GET | `/api/memory/user/{user_id}` | 用户全部历史计划 |
| POST | `/api/route` | 高德路线规划 |
| GET | `/api/weather/{city}` | 天气查询 |

---

## 目录结构

```
LangChain/
├── web/                      # FastAPI 与前端
│   ├── main.py
│   └── static/index.html
├── app/
│   ├── agents/               # ReAct / LangGraph / Dialog / Version
│   ├── memory/               # Redis 长期记忆（TravelMemoryManager）
│   ├── rag/                  # Hybrid RAG、向量库、Rerank
│   ├── services/             # 高德、LLM 等服务
│   ├── tools/                # Agent 工具（景点/天气/酒店/路线/RAG）
│   ├── prompts/              # 渐进式披露 md 提示词
│   ├── models/               # Pydantic 数据模型
│   ├── knowledge_base/       # 多城市景点知识库
│   ├── kg/                   # Neo4j 图检索（可选）
│   ├── monitoring/           # LangSmith / 指标
│   └── data/                 # 文件回退存储（计划、对话会话）
├── scripts/                  # 评测与联调脚本
├── requirements.txt
└── .env.example
```

---

## 设计亮点

- **记忆驱动推荐**：Redis 长期记忆让系统「记得你去过哪」，减少重复景点、支持同城差异化规划  
- **知识库 + 实时地图**：RAG 负责景点质量与细节，高德负责坐标、天气与真实路况耗时  
- **可降级**：Redis / 高德 / 外部观测均可降级，保证本地可演示、可开发  
- **双编排路径**：线上 Web 用 ReAct 多智能体；LangGraph 提供并行与反思，便于后续切换或研究

---

## 注意事项

1. 请妥善保管 `.env` 中的 API Key，勿提交到公开仓库  
2. 首次使用需保证 Qdrant 中已有景点向量数据（或按知识库脚本导入）  
3. 启用 Redis 后，重启服务即可自动写入长期记忆；未启动 Redis 时会打印回退提示并使用本地文件
