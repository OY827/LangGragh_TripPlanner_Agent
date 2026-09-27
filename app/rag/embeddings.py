"""Embedding 工厂：DashScope / HuggingFace"""

import os
from typing import Any, Optional, Tuple


def build_embeddings(
    embedding_backend: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
) -> Tuple[Any, str, str]:
    """创建 Embedding 实例，返回 (embeddings, backend, model_name)。"""
    backend = (embedding_backend or os.getenv("EMBEDDING_BACKEND", "dashscope")).lower()

    if backend == "huggingface":
        model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
        try:
            embeddings = _load_huggingface_embeddings(model_name, local_files_only=True)
            print(f"✅ 使用 HuggingFace / SentenceTransformer Embedding: {model_name}")
            return embeddings, "huggingface", model_name
        except Exception as e:
            print(f"[warn] local HuggingFace model unavailable, fallback DashScope: {e}")
            backend = "dashscope"
            embedding_model_name = None

    dashscope_cls = _load_dashscope_embeddings()
    if dashscope_cls is not None:
        model_name = (
            embedding_model_name
            if embedding_model_name and not str(embedding_model_name).startswith("BAAI/")
            else os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v1")
        )
        embeddings = dashscope_cls(
            model=model_name,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        print(f"✅ 使用阿里云 DashScope Embedding: {model_name}")
        return embeddings, "dashscope", model_name

    model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    embeddings = _load_huggingface_embeddings(model_name, local_files_only=False)
    print(f"✅ 使用 HuggingFace / SentenceTransformer Embedding: {model_name}")
    return embeddings, "huggingface", model_name


def _load_dashscope_embeddings():
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings
    except ImportError:
        try:
            from langchain_dashscope import DashScopeEmbeddings
            return DashScopeEmbeddings
        except ImportError:
            return None


def _load_huggingface_embeddings(model_name: str, local_files_only: bool = False):
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model_name, local_files_only=local_files_only)

    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    except Exception as e:
        print(f"⚠️ HuggingFaceEmbeddings 初始化失败: {e}")
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
