"""
系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    # 原始食谱markdown存放路径
    data_path: str = "data/C8/cook"
    # 索引保存路径：向量数据库文件存放位置
    index_save_path: str = "./vector_index"

    embedding_model: str = "text-embedding-v2"
    llm_model: str = "qwen-plus"

    top_k: int = 3
    temperature: float = 0.1
    max_tokens: int = 2048

    # __post_init__是dataclass的特殊方法，会在__init__之后自动调用
    def __post_init__(self):
        pass # pass表示什么也不做

    # 类方法：允许我们用一个字典来创建RAGCofig实例
    #   比如从JSON文件读取配置后，直接用dict生成配置对象
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """从字典创建配置对象"""
        return cls(**config_dict)

    # 将配置对象转回字典，方便保存到文件或打印
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

# 创建一个默认的配置实例，供其他模块直接导入使用
# 这样就不用每次都在各处重复写 RAGConfig() 了
DEFAULT_CONFIG = RAGConfig()