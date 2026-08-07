"""
向量存储模块 - Vector Store
基于 Qdrant 的向量存储实现
支持本地 HuggingFace 模型和阿里云 Embedding API
支持向量检索 + BM25 混合检索
"""

import os
import re
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore as LangChainQdrant

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    print("⚠️ rank_bm25 未安装，BM25检索不可用")

_embedding_backend = os.getenv("EMBEDDING_BACKEND", "dashscope").lower()

if _embedding_backend == "dashscope":
    try:
        from langchain_dashscope import DashScopeEmbeddings
    except ImportError:
        _embedding_backend = "huggingface"
        print("⚠️ langchain-dashscope 未安装，使用 HuggingFace Embeddings")

if _embedding_backend == "huggingface":
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None
    
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
    except ImportError:
        SentenceTransformerEmbeddings = None


class QdrantVectorStore:
    """基于LangChain的Qdrant向量存储管理器"""

    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        embedding_model_name: str = None,
        use_memory: bool = False,
        embedding_backend: str = None
    ):
        # 优先使用环境变量配置
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        # 根据配置选择连接方式
        if use_memory:
            self.client = QdrantClient(":memory:", api_key=self.api_key)
        else:
            self.client = QdrantClient(
                url=self.url, 
                api_key=self.api_key,
                timeout=60,
                check_compatibility=False
            )

        # 根据后端选择Embedding模型
        backend = embedding_backend or _embedding_backend
        
        if backend == "dashscope":
            model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
            self.embeddings = DashScopeEmbeddings(
                model=model_name,
                api_key=os.getenv("DASHSCOPE_API_KEY")
            )
            print(f"✅ 使用阿里云 DashScope Embedding: {model_name}")
        else:
            model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
            if HuggingFaceEmbeddings is not None:
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True}
                    )
                    print(f"✅ 使用 HuggingFace Embedding: {model_name}")
                except Exception as e:
                    print(f"⚠️ HuggingFaceEmbeddings 初始化失败: {e}")
                    if SentenceTransformerEmbeddings is not None:
                        self.embeddings = SentenceTransformerEmbeddings(
                            model_name=model_name,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True}
                        )
                        print(f"✅ 使用 SentenceTransformer Embedding: {model_name}")
                    else:
                        raise ImportError("sentence-transformers 未安装，请运行: pip install sentence-transformers")
            elif SentenceTransformerEmbeddings is not None:
                self.embeddings = SentenceTransformerEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                print(f"✅ 使用 SentenceTransformer Embedding: {model_name}")
            else:
                raise ImportError("sentence-transformers 未安装，请运行: pip install sentence-transformers")

        # 默认集合名称
        self.default_collection = os.getenv("QDRANT_COLLECTION", "travel_knowledge")
        
        # BM25索引缓存
        self._bm25_indexes = {}
        self._bm25_corpus = {}
    
    def _create_collection_if_not_exists(self, collection_name: str) -> None:
        """如果集合不存在则创建"""
        if not self.client.collection_exists(collection_name=collection_name):
            print(f"创建集合 '{collection_name}'...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self._get_embedding_dimension(),
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0
                ),
                on_disk_payload=True
            )
            
            # 创建关键字索引用于过滤
            self._create_keyword_index(collection_name)
        else:
            print(f"集合 '{collection_name}' 已存在")
    
    def _create_keyword_index(self, collection_name: str) -> None:
        """为过滤字段创建关键字索引"""
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="city",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer="word"
                )
            )
            print(f"✅ 已为 'city' 字段创建索引")
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="category",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer="word"
                )
            )
            print(f"✅ 已为 'category' 字段创建索引")
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="title",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer="word"
                )
            )
            print(f"✅ 已为 'title' 字段创建索引")
        except Exception as e:
            print(f"⚠️ 创建索引失败: {e}")
    
    def _get_embedding_dimension(self) -> int:
        """获取Embedding模型的维度"""
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)
    
    def get_store(self, collection_name: str = None) -> LangChainQdrant:
        """获取或创建Qdrant向量存储实例"""
        collection = collection_name or self.default_collection
        self._create_collection_if_not_exists(collection)

        return LangChainQdrant(
            client=self.client,
            collection_name=collection,
            embedding=self.embeddings
        )
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str = None
    ) -> None:
        """向向量库添加文档"""
        collection = collection_name or self.default_collection
        self._create_collection_if_not_exists(collection)
        
        # 转换为LangChain Document格式
        langchain_docs = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            langchain_docs.append(Document(
                page_content=doc["content"],
                metadata=metadata
            ))
        
        qdrant = self.get_store(collection)
        qdrant.add_documents(langchain_docs)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] = None,
        collection_name: str = None
    ) -> None:
        """向向量库添加文本"""
        collection = collection_name or self.default_collection
        self._create_collection_if_not_exists(collection)
        
        qdrant = self.get_store(collection)
        qdrant.add_texts(texts=texts, metadatas=metadatas)
    
    def search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        k: int = 5,
        collection_name: str = None
    ) -> List[Dict[str, Any]]:
        """检索相关文档（使用本地过滤，增加候选数量）"""
        collection = collection_name or self.default_collection
        
        if not self.client.collection_exists(collection_name=collection):
            return []
        
        qdrant = self.get_store(collection)
        
        # 增加候选数量，确保跨城市检索时能获取到足够结果
        fetch_k = k * 50
        
        results = qdrant.similarity_search_with_score(query=query, k=fetch_k)
        
        # 在本地进行过滤
        if filters:
            results = [
                (doc, score) for doc, score in results
                if self._matches_filters(doc.metadata, filters)
            ]
        
        # 返回前k个结果
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results[:k]
        ]
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查文档是否匹配过滤条件"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def _build_filter(self, filters: Dict[str, Any]) -> models.Filter:
        """构建Qdrant过滤条件"""
        must_conditions = []

        for key, value in filters.items():
            if isinstance(value, str):
                must_conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                ))
            elif isinstance(value, list):
                must_conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value)
                ))
            elif isinstance(value, (int, float)):
                must_conditions.append(models.FieldCondition(
                    key=key,
                    range=models.Range(gte=value)
                ))

        return models.Filter(must=must_conditions)
    
    def delete_collection(self, collection_name: str = None) -> None:
        """删除指定集合"""
        collection = collection_name or self.default_collection
        if self.client.collection_exists(collection_name=collection):
            self.client.delete_collection(collection_name=collection)
    
    def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """获取集合统计信息"""
        collection = collection_name or self.default_collection
        
        if not self.client.collection_exists(collection_name=collection):
            return {"error": "Collection not found"}
        
        info = self.client.get_collection(collection_name=collection)
        info_dict = info.model_dump()
        
        return {
            "collection_name": collection,
            "vector_count": info_dict.get("points_count", info_dict.get("indexed_vectors_count", 0)),
            "dimension": info_dict.get("config", {}).get("params", {}).get("vectors", {}).get("size", 0),
            "distance": str(info_dict.get("config", {}).get("params", {}).get("vectors", {}).get("distance", ""))
        }
    
    def as_retriever(self, collection_name: str = None, **kwargs):
        """获取检索器对象"""
        collection = collection_name or self.default_collection
        qdrant = self.get_store(collection)
        return qdrant.as_retriever(**kwargs)
    
    def _build_bm25_index(self, collection_name: str = None) -> None:
        """构建BM25索引"""
        if BM25Okapi is None:
            return
        
        collection = collection_name or self.default_collection
        
        if collection in self._bm25_indexes:
            return
        
        try:
            qdrant = self.get_store(collection)
            
            all_docs = []
            offset = 0
            batch_size = 100
            
            while True:
                results = qdrant.similarity_search_with_score(
                    query="",
                    k=batch_size,
                    offset=offset
                )
                if not results:
                    break
                all_docs.extend(results)
                offset += batch_size
                if len(results) < batch_size:
                    break
            
            corpus = []
            doc_list = []
            
            for doc, score in all_docs:
                text = doc.page_content
                if text:
                    tokens = self._tokenize(text)
                    corpus.append(tokens)
                    doc_list.append((doc, score))
            
            if corpus:
                bm25 = BM25Okapi(corpus)
                self._bm25_indexes[collection] = bm25
                self._bm25_corpus[collection] = doc_list
                print(f"✅ BM25索引构建完成，共{len(corpus)}条文档")
        except Exception as e:
            print(f"⚠️ BM25索引构建失败: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """中文文本分词"""
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        words = text.split()
        result = []
        for word in words:
            if word:
                if re.match(r'^[\u4e00-\u9fff]+$', word):
                    result.extend(list(word))
                else:
                    result.append(word)
        return result
    
    def bm25_search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        k: int = 5,
        collection_name: str = None
    ) -> List[Dict[str, Any]]:
        """BM25关键词检索"""
        if BM25Okapi is None:
            return self.search(query, filters, k, collection_name)
        
        collection = collection_name or self.default_collection
        
        self._build_bm25_index(collection)
        
        if collection not in self._bm25_indexes:
            return self.search(query, filters, k, collection_name)
        
        bm25 = self._bm25_indexes[collection]
        doc_list = self._bm25_corpus[collection]
        
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)
        
        scored_docs = sorted(zip(doc_list, scores), key=lambda x: x[1], reverse=True)
        
        results = []
        for (doc, _), score in scored_docs[:k]:
            if filters and not self._matches_filters(doc.metadata, filters):
                continue
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        return results[:k]
    
    def hybrid_search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        k: int = 5,
        collection_name: str = None,
        vector_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        混合检索：向量检索 + BM25
        
        Args:
            query: 查询文本
            filters: 过滤条件
            k: 返回结果数量
            collection_name: 集合名称
            vector_weight: 向量检索权重（0-1），BM25权重 = 1 - vector_weight
        """
        collection = collection_name or self.default_collection
        
        vector_results = self.search(query, filters, k * 3, collection)
        
        bm25_results = self.bm25_search(query, filters, k * 3, collection)
        
        scores = {}
        seen = {}
        
        if vector_results:
            max_vector = max(r.get("score", 0) for r in vector_results) or 1
            for r in vector_results:
                key = r.get("metadata", {}).get("title", r.get("content", "")[:50])
                norm_score = r.get("score", 0) / max_vector
                scores[key] = scores.get(key, 0) + norm_score * vector_weight
                seen[key] = r
        
        if bm25_results:
            max_bm25 = max(r.get("score", 0) for r in bm25_results) or 1
            for r in bm25_results:
                key = r.get("metadata", {}).get("title", r.get("content", "")[:50])
                norm_score = r.get("score", 0) / max_bm25
                scores[key] = scores.get(key, 0) + norm_score * (1 - vector_weight)
                if key not in seen:
                    seen[key] = r
        
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [seen[k] for k in sorted_keys[:k] if k in seen]