"""
文档切分器模块 - Document Splitter
负责将长文档切分为适合向量化的小块

核心功能：
1. 支持多种切分策略（字符、Token、句子）
2. 支持中英文混合文本
3. 保持语义完整性
"""

from typing import List, Dict, Optional, Union
from enum import Enum
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SplitStrategy(Enum):
    """切分策略枚举"""
    CHARACTER = "character"          # 基于字符数切分
    TOKEN = "token"                  # 基于Token数切分
    SENTENCE = "sentence"            # 基于句子切分
    SEMANTIC = "semantic"            # 基于语义切分（保留段落）


class DocumentSplitter:
    """
    文档切分器
    
    支持多种切分策略，适用于不同类型的文档。
    """

    def __init__(
        self,
        strategy: Union[str, SplitStrategy] = "character",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        初始化切分器
        
        Args:
            strategy: 切分策略 ("character", "token", "sentence", "semantic")
            chunk_size: 每个块的大小（字符数或Token数）
            chunk_overlap: 块之间的重叠大小
            model_name: 用于Token计算的模型名称
        """
        if isinstance(strategy, str):
            strategy = SplitStrategy(strategy)
        
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 根据策略初始化切分器
        self.splitter = self._init_splitter(strategy, model_name)

    def _init_splitter(self, strategy: SplitStrategy, model_name: str):
        """根据策略初始化切分器"""
        if strategy == SplitStrategy.CHARACTER:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n", "\n", "。", "！", "？", ".", "!", "?",
                    "；", ";", "，", ",", "、", " "
                ]
            )
        else:
            # 默认使用字符切分
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n", "\n\n\n",  # 优先按段落切分
                    "\n",              # 然后按换行切分
                    "。", "！", "？",   # 中文句子结尾
                    ".", "!", "?",     # 英文句子结尾
                    "；", ";",          # 分号
                    "，", ",",          # 逗号
                    "、",               # 顿号
                    " "                # 空格
                ]
            )

    def split_text(self, text: str) -> List[str]:
        """
        切分文本为块列表
        
        Args:
            text: 待切分的文本
            
        Returns:
            切分后的块列表
        """
        if not text or not isinstance(text, str):
            return []
        
        # 预处理文本
        text = self._preprocess_text(text)
        
        # 使用对应的切分器
        if self.strategy in [SplitStrategy.CHARACTER, SplitStrategy.SEMANTIC]:
            return self.splitter.split_text(text)
        elif self.strategy == SplitStrategy.TOKEN:
            return self.splitter.split_text(text)
        elif self.strategy == SplitStrategy.SENTENCE:
            return self.splitter.split_text(text)
        
        return [text]

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        切分文档列表
        
        Args:
            documents: 文档列表，每个文档包含 "content" 和 "metadata" 字段
            
        Returns:
            切分后的文档块列表
        """
        chunks = []
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # 切分内容
            text_chunks = self.split_text(content)
            
            # 创建块
            for i, chunk in enumerate(text_chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "original_length": len(content),
                    "chunk_length": len(chunk)
                }
                
                chunks.append({
                    "content": chunk,
                    "metadata": chunk_metadata
                })
        
        return chunks

    def split_with_overlap_control(self, text: str, target_size: int = None) -> List[str]:
        """
        智能切分：根据目标大小动态调整
        
        Args:
            text: 待切分的文本
            target_size: 目标块大小（可选）
            
        Returns:
            切分后的块列表
        """
        if not text:
            return []
        
        target_size = target_size or self.chunk_size
        text_length = len(text)
        
        # 如果文本长度小于目标大小，直接返回
        if text_length <= target_size:
            return [text]
        
        # 计算需要的块数
        num_chunks = (text_length + target_size - 1) // target_size
        
        # 计算实际的重叠大小
        actual_overlap = (num_chunks * target_size - text_length) // (num_chunks - 1) if num_chunks > 1 else 0
        
        # 创建临时切分器
        temp_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_size,
            chunk_overlap=actual_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ","]
        )
        
        return temp_splitter.split_text(text)

    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text:
            return ""
        
        # 去除多余空白
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        
        # 合并连续的换行
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text

    def get_chunk_info(self, text: str) -> Dict:
        """
        获取文本切分信息
        
        Args:
            text: 文本
            
        Returns:
            切分信息字典
        """
        chunks = self.split_text(text)
        
        return {
            "original_length": len(text),
            "num_chunks": len(chunks),
            "avg_chunk_length": sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
            "max_chunk_length": max(len(c) for c in chunks) if chunks else 0,
            "min_chunk_length": min(len(c) for c in chunks) if chunks else 0
        }


# ==================== 便捷函数 ====================

def create_splitter(
    strategy: str = "character",
    **kwargs
) -> DocumentSplitter:
    """创建文档切分器实例"""
    return DocumentSplitter(strategy=strategy, **kwargs)


def split_text_by_paragraph(text: str, max_length: int = 1000) -> List[str]:
    """
    按段落切分文本
    
    Args:
        text: 文本
        max_length: 最大段落长度
        
    Returns:
        段落列表
    """
    # 按双换行符切分段落
    paragraphs = text.split("\n\n")
    
    # 处理过长的段落
    result = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(para) <= max_length:
            result.append(para)
        else:
            # 切分过长的段落
            splitter = DocumentSplitter(strategy="character", chunk_size=max_length, chunk_overlap=50)
            result.extend(splitter.split_text(para))
    
    return result


def split_text_by_sentence(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """
    按句子切分文本
    
    Args:
        text: 文本
        sentences_per_chunk: 每块包含的句子数
        
    Returns:
        句子块列表
    """
    import re
    
    # 分割句子（支持中英文）
    sentence_endings = r'(?<=[。！？.!?])\s+'
    sentences = re.split(sentence_endings, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 组合句子
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        
        if len(current_chunk) >= sentences_per_chunk:
            chunks.append("。".join(current_chunk) + "。")
            current_chunk = []
    
    if current_chunk:
        chunks.append("。".join(current_chunk) + ("。" if text.endswith(("。", "！", "？")) else ""))
    
    return chunks


# ==================== 示例 ====================

if __name__ == "__main__":
    # 示例文本
    sample_text = """
故宫是明清两代的皇家宫殿，位于北京市中心。始建于1406年，历时14年建成，是世界上现存规模最大、保存最为完整的木质结构古建筑群。

故宫占地面积约72万平方米，建筑面积约15万平方米，有宫殿9999间半。著名景点包括太和殿、中和殿、保和殿、乾清宫、交泰殿、坤宁宫等。

太和殿是故宫中最大的宫殿，高35.05米，面积2377平方米，是举行重大典礼的场所。中和殿是皇帝在大典前休息的地方，保和殿则是科举考试殿试的地点。

建议游览时间：3-4小时。门票价格：旺季60元，淡季40元。
"""

    # 测试不同切分策略
    print("=" * 60)
    print("📝 文档切分器测试")
    print("=" * 60)

    strategies = ["character", "token", "sentence", "semantic"]
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} 切分 ---")
        splitter = DocumentSplitter(strategy=strategy, chunk_size=200, chunk_overlap=30)
        chunks = splitter.split_text(sample_text)
        
        info = splitter.get_chunk_info(sample_text)
        print(f"原始长度: {info['original_length']}, 块数: {info['num_chunks']}")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n块 {i}/{len(chunks)} ({len(chunk)}字符):")
            print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)

    # 测试文档列表切分
    print("\n--- 文档列表切分 ---")
    documents = [
        {
            "content": sample_text,
            "metadata": {"title": "故宫介绍", "city": "北京"}
        }
    ]
    
    splitter = DocumentSplitter(strategy="semantic", chunk_size=200)
    chunks = splitter.split_documents(documents)
    
    for chunk in chunks:
        print(f"\n标题: {chunk['metadata']['title']}")
        print(f"块索引: {chunk['metadata']['chunk_index']}/{chunk['metadata']['total_chunks']}")
        print(f"内容: {chunk['content'][:80]}...")