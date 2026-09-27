"""
本地 Chroma 向量存储
接口与 QdrantVectorStore 对齐：写入、过滤检索、BM25 混合检索
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    print("⚠️ rank_bm25 未安装，BM25检索不可用")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.rag.embeddings import build_embeddings


def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Chroma 只接受 str / int / float / bool，嵌套结构需展平。"""
    if not metadata:
        return {}

    out: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if key == "location" and isinstance(value, dict):
            lon = value.get("longitude")
            lat = value.get("latitude")
            if lon is not None:
                try:
                    out["longitude"] = float(lon)
                except (TypeError, ValueError):
                    pass
            if lat is not None:
                try:
                    out["latitude"] = float(lat)
                except (TypeError, ValueError):
                    pass
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float, str)):
            out[key] = value
        elif isinstance(value, list):
            out[key] = json.dumps(value, ensure_ascii=False) if value else ""
        elif isinstance(value, dict):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = str(value)
    return out


def _prepare_content(content: str, metadata: Dict[str, Any]) -> str:
    """DashScope embedding 输入长度限制为 2048，超长文本截断后再向量化。"""
    max_chars = int(os.getenv("CHROMA_EMBED_MAX_CHARS", "2000"))
    text = (content or "").strip() or str(metadata.get("title") or "景点")
    if len(text) > max_chars:
        metadata["content_full"] = text
        return text[:max_chars]
    return text


def _restore_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    md = dict(metadata or {})
    if "location" not in md and "longitude" in md and "latitude" in md:
        try:
            md["location"] = {
                "longitude": float(md["longitude"]),
                "latitude": float(md["latitude"]),
            }
        except (TypeError, ValueError):
            pass
    return md


def _restore_document(content: str, metadata: Optional[Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
    md = _restore_metadata(metadata)
    full = md.pop("content_full", None)
    return (full or content), md


def _doc_id(content: str, metadata: Dict[str, Any]) -> str:
    key = "|".join(
        [
            str(metadata.get("city", "")),
            str(metadata.get("doc_type", "")),
            str(metadata.get("section", "")),
            str(metadata.get("title", "")),
            str(metadata.get("chunk_index", "")),
            content[:80],
        ]
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


class ChromaVectorStore:
    """基于本地持久化 Chroma 的向量存储管理器。"""

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embedding_model_name: str = None,
        embedding_backend: str = None,
        **_ignored,
    ):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_path = os.path.join(project_root, "app", "data", "chroma")
        self.persist_directory = persist_directory or os.getenv("CHROMA_PATH", default_path)
        os.makedirs(self.persist_directory, exist_ok=True)

        import chromadb

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        chroma_backend = embedding_backend or os.getenv("CHROMA_EMBEDDING_BACKEND") or "dashscope"
        self.embeddings, _, _ = build_embeddings(chroma_backend, embedding_model_name)
        self.default_collection = (
            collection_name
            or os.getenv("CHROMA_COLLECTION")
            or os.getenv("QDRANT_COLLECTION", "travel_knowledge")
        )
        self._stores: Dict[str, Any] = {}
        self._bm25_indexes: Dict[str, Any] = {}
        self._bm25_corpus: Dict[str, List] = {}
        print(f"✅ 使用本地 Chroma: {self.persist_directory} / {self.default_collection}")

    def collection_exists(self, collection_name: str = None) -> bool:
        collection = collection_name or self.default_collection
        names = {item.name for item in self.client.list_collections()}
        return collection in names

    def get_store(self, collection_name: str = None):
        from langchain_chroma import Chroma

        collection = collection_name or self.default_collection
        if collection not in self._stores:
            self._stores[collection] = Chroma(
                client=self.client,
                collection_name=collection,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"},
                create_collection_if_not_exists=True,
            )
        return self._stores[collection]

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str = None,
    ) -> None:
        collection = collection_name or self.default_collection
        store = self.get_store(collection)

        langchain_docs = []
        ids = []
        for doc in documents:
            metadata = _sanitize_metadata(doc.get("metadata", {}))
            content = _prepare_content(doc.get("content") or "", metadata)
            langchain_docs.append(Document(page_content=content, metadata=metadata))
            ids.append(_doc_id(doc.get("content") or content, metadata))

        batch_size = int(os.getenv("CHROMA_UPSERT_BATCH", "32"))
        total = len(langchain_docs)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            store.add_documents(langchain_docs[start:end], ids=ids[start:end])
            print(f"  Chroma 写入 {end}/{total}")

        self._bm25_indexes.pop(collection, None)
        self._bm25_corpus.pop(collection, None)

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] = None,
        collection_name: str = None,
    ) -> None:
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            docs.append({"content": text, "metadata": meta})
        self.add_documents(docs, collection_name)

    def search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        k: int = 5,
        collection_name: str = None,
    ) -> List[Dict[str, Any]]:
        collection = collection_name or self.default_collection
        if not self.collection_exists(collection):
            return []

        store = self.get_store(collection)
        where = self._build_where(filters)
        fetch_k = k if where else min(max(k * 3, k), 50)

        kwargs: Dict[str, Any] = {"query": query, "k": fetch_k}
        if where:
            kwargs["filter"] = where

        try:
            results = store.similarity_search_with_score(**kwargs)
        except Exception:
            results = store.similarity_search_with_score(query=query, k=fetch_k)
            if filters:
                results = [
                    (doc, score) for doc, score in results
                    if self._matches_filters(_restore_metadata(doc.metadata), filters)
                ]

        formatted = []
        for doc, score in results[:k]:
            content, metadata = _restore_document(doc.page_content, doc.metadata)
            formatted.append({
                "content": content,
                "metadata": metadata,
                "score": self._to_similarity(score),
            })
        return formatted

    def _to_similarity(self, score: float) -> float:
        """Chroma cosine 距离越小越相似，转成 0-1 相似度供后续融合/阈值使用。"""
        distance = float(score)
        if 0.0 <= distance <= 2.0:
            return max(0.0, 1.0 - distance)
        return distance

    def _build_where(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def delete_collection(self, collection_name: str = None) -> None:
        collection = collection_name or self.default_collection
        if self.collection_exists(collection):
            self.client.delete_collection(collection)
        self._stores.pop(collection, None)
        self._bm25_indexes.pop(collection, None)
        self._bm25_corpus.pop(collection, None)

    def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        collection = collection_name or self.default_collection
        if not self.collection_exists(collection):
            return {"error": "Collection not found"}

        count = self.client.get_collection(collection).count()
        dimension = 0
        try:
            dimension = len(self.embeddings.embed_query("test"))
        except Exception:
            pass

        return {
            "collection_name": collection,
            "vector_count": count,
            "dimension": dimension,
            "distance": "cosine",
            "backend": "chroma",
            "persist_directory": self.persist_directory,
        }

    def as_retriever(self, collection_name: str = None, **kwargs):
        return self.get_store(collection_name).as_retriever(**kwargs)

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        words = text.split()
        result = []
        for word in words:
            if not word:
                continue
            if re.match(r"^[\u4e00-\u9fff]+$", word):
                result.extend(list(word))
            else:
                result.append(word)
        return result

    def _build_bm25_index(self, collection_name: str = None) -> None:
        if BM25Okapi is None:
            return

        collection = collection_name or self.default_collection
        if collection in self._bm25_indexes:
            return
        if not self.collection_exists(collection):
            return

        try:
            raw = self.client.get_collection(collection).get(include=["documents", "metadatas"])
            corpus = []
            doc_list = []
            documents = raw.get("documents") or []
            metadatas = raw.get("metadatas") or []
            for text, metadata in zip(documents, metadatas):
                if not text:
                    continue
                content, restored = _restore_document(text, metadata)
                doc = Document(page_content=content, metadata=restored)
                corpus.append(self._tokenize(content))
                doc_list.append((doc, 0.0))

            if corpus:
                self._bm25_indexes[collection] = BM25Okapi(corpus)
                self._bm25_corpus[collection] = doc_list
                print(f"✅ BM25索引构建完成，共{len(corpus)}条文档")
        except Exception as e:
            print(f"⚠️ BM25索引构建失败: {e}")

    def bm25_search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        k: int = 5,
        collection_name: str = None,
    ) -> List[Dict[str, Any]]:
        if BM25Okapi is None:
            return self.search(query, filters, k, collection_name)

        collection = collection_name or self.default_collection
        self._build_bm25_index(collection)
        if collection not in self._bm25_indexes:
            return self.search(query, filters, k, collection_name)

        bm25 = self._bm25_indexes[collection]
        doc_list = self._bm25_corpus[collection]
        scores = bm25.get_scores(self._tokenize(query))
        scored_docs = sorted(zip(doc_list, scores), key=lambda x: x[1], reverse=True)

        results = []
        for (doc, _), score in scored_docs:
            metadata = _restore_metadata(doc.metadata)
            if filters and not self._matches_filters(metadata, filters):
                continue
            results.append({
                "content": doc.page_content,
                "metadata": metadata,
                "score": float(score),
            })
            if len(results) >= k:
                break
        return results

    def hybrid_search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        k: int = 5,
        collection_name: str = None,
        vector_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        collection = collection_name or self.default_collection
        vector_results = self.search(query, filters, k * 3, collection)
        bm25_results = self.bm25_search(query, filters, k * 3, collection)

        scores: Dict[str, float] = {}
        seen: Dict[str, Dict[str, Any]] = {}

        if vector_results:
            max_vector = max(r.get("score", 0) for r in vector_results) or 1
            for result in vector_results:
                key = result.get("metadata", {}).get("title", result.get("content", "")[:50])
                scores[key] = scores.get(key, 0) + (result.get("score", 0) / max_vector) * vector_weight
                seen[key] = result

        if bm25_results:
            max_bm25 = max(r.get("score", 0) for r in bm25_results) or 1
            for result in bm25_results:
                key = result.get("metadata", {}).get("title", result.get("content", "")[:50])
                scores[key] = scores.get(key, 0) + (result.get("score", 0) / max_bm25) * (1 - vector_weight)
                if key not in seen:
                    seen[key] = result

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [seen[key] for key in sorted_keys[:k] if key in seen]
