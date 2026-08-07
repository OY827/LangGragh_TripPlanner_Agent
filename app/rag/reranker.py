"""
Rerank 模块 - CrossEncoder 精排

使用 CrossEncoder 模型对初步检索的文档进行精排，提升 Top-K 结果的相关性。
CrossEncoder 将 query 和 document 拼接后输入模型，输出相关性分数，
比向量检索的余弦相似度更精确。

支持的模型：
- cross-encoder/ms-marco-MiniLM-L-6-v2 (默认，轻量快速)
- BAAI/bge-reranker-base (中文效果好)
- BAAI/bge-reranker-large (精度更高，但更慢)
"""

import os
from typing import List, Dict, Any, Optional
import importlib.util


class Reranker:
    """CrossEncoder 重排器"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        use_fp16: bool = True,
        device: str = None
    ):
        """
        初始化重排器

        Args:
            model_name: 模型名称，优先使用中文优化模型
            use_fp16: 是否使用半精度加速
            device: 指定设备 (cpu/cuda)，None则自动选择
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self._model = None
        self._tokenizer = None
        self._available = False

        self._try_load_model()

    def _try_load_model(self):
        """尝试加载模型，带优雅降级"""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512
            )
            self._available = True
        except ImportError:
            print("⚠️ sentence-transformers 未安装，Rerank 使用降级模式（LLM打分）")
            self._available = False
        except Exception as e:
            print(f"⚠️ CrossEncoder 模型加载失败: {e}")
            print("   Rerank 降级为 LLM 打分模式")
            self._available = False

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 10,
        content_field: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        对文档列表进行重排

        Args:
            query: 用户查询
            docs: 待重排的文档列表
            top_k: 返回前K个文档
            content_field: 文档内容字段名

        Returns:
            重排后的文档列表（附带 rerank_score）
        """
        if not docs:
            return []

        # 即使候选数 ≤ top_k，也要重排打分，保证排序质量
        if self._available:
            return self._rerank_with_cross_encoder(query, docs, top_k, content_field)
        else:
            return self._rerank_with_llm_fallback(query, docs, top_k, content_field)

    def _rerank_with_cross_encoder(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int,
        content_field: str
    ) -> List[Dict[str, Any]]:
        """使用 CrossEncoder 进行精排"""
        try:
            # 构建 query-doc 对：把标题/分类/标签前置，增强意图信号
            pairs = []
            for doc in docs:
                content = self._build_rerank_doc_text(doc, content_field)
                pairs.append((query, content))

            # 批量计算相关性分数
            scores = self._model.predict(pairs)

            # 将分数附加到文档
            for doc, score in zip(docs, scores):
                doc["rerank_score"] = float(score)

            # 按 rerank_score 降序排序
            ranked_docs = sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)

            return ranked_docs[:top_k]

        except Exception as e:
            print(f"⚠️ CrossEncoder 重排失败: {e}，降级为LLM打分")
            return self._rerank_with_llm_fallback(query, docs, top_k, content_field)

    def _build_rerank_doc_text(self, doc: Dict[str, Any], content_field: str) -> str:
        """拼接更利于意图识别的文档文本"""
        meta = doc.get("metadata") or {}
        title = meta.get("title") or ""
        category = meta.get("category") or ""
        tags = meta.get("tags") or ""
        if isinstance(tags, list):
            tags = "、".join(str(t) for t in tags)
        content = doc.get(content_field, "") or ""
        header = f"景点名称：{title}。分类：{category}。标签：{tags}。"
        text = f"{header}\n{content}"
        return text[:512]

    def _rerank_with_llm_fallback(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int,
        content_field: str
    ) -> List[Dict[str, Any]]:
        """LLM 打分降级方案：使用原始检索分数 + 简单关键词匹配"""
        import re

        # 提取查询关键词
        query_keywords = set(re.findall(r'[\u4e00-\u9fff]+', query))

        for doc in docs:
            content = doc.get(content_field, "")
            title = doc.get("metadata", {}).get("title", "")
            combined_text = f"{title} {content}"

            # 关键词匹配得分
            keyword_matches = sum(1 for kw in query_keywords if kw in combined_text)
            keyword_score = keyword_matches / max(len(query_keywords), 1)

            # 原始检索分数
            original_score = doc.get("score", 0)

            # 综合分数
            doc["rerank_score"] = float(original_score * 0.6 + keyword_score * 0.4)

        ranked_docs = sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return ranked_docs[:top_k]

    @property
    def is_available(self) -> bool:
        """是否使用 CrossEncoder 模型"""
        return self._available

    def get_info(self) -> Dict[str, Any]:
        """获取重排器信息"""
        return {
            "model_name": self.model_name,
            "mode": "cross_encoder" if self._available else "llm_fallback",
            "available": self._available
        }
