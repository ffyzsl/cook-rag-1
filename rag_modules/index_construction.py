import logging
import os
from typing import List
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块- 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str="./vector_index"):
        self.model_name = model_name # 保存嵌入模型名称，后续用来创建DashScopeEmbeddings
        self.index_save_path = index_save_path # 索引文件的保存目录
        self.embeddings = None # 存放一个“embeddings”对象，用来将任意文字转化为向量
        self.vectorstore = None # 用来存放FAISS向量存储实例
        self.setup_embeddings()

    def setup_embeddings(self):
        """
        初始化嵌入模型
        :return:
        """
        logger.info(f"正在初始化阿里云百炼嵌入模型: {self.model_name}")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY. 请在.env文件中设置")

        # 初始化 DashScope embeddings
        self.embeddings = DashScopeEmbeddings(
            model = self.model_name,
            dashscope_api_key= api_key,
        )

        logger.info("阿里云百炼平台嵌入模型初始化完成")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        接受文档列表，将其向量化并构建FAISS索引
        :param chunks:
        :return: 返回向量存储对象
        """

        logger.info("正在构建FAISS向量索引...")
        if not chunks:
            raise ValueError("文档块列表不能为空")

        # 构建FAISS向量存储
        # FAISS.from_documents(documents=chunks, embedding=self.embeddings)：
        # LangChain 的这个方法会自动做两件事：
        # 对 chunks 中的每个文档，调用 self.embeddings.embed_query() 或 embed_documents() 将其文本转为向量。
        # 将所有向量建立 FAISS 索引。
        # 因为这个过程会自动 embedding 对象，所以只要你保证了 self.embeddings 正确地指向了阿里云百炼，这部分就完全不用动。
        self.vectorstore = FAISS.from_documents(
            documents = chunks,
            embedding = self.embeddings,
        )

        logger.info(f"向量索引构建完成，包含{len(chunks)}个向量")
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        """
        接受文档块列表，将其向量化并构建FAISS索引
        :param new_chunks:
        :return: 返回向量存储对象
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        logger.info(f"正在添加{len(new_chunks)}个新文档到索引...")
        # 调用 self.vectorstore.add_documents(new_chunks)：该方法同样会使用 self.embeddings 对新文档进行向量化，并将新向量追加到现有的 FAISS 索引中。
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档添加完成")

    def save_index(self):
        """
        将当前内存中的FAISS索引持久化保存到磁盘中
        :return:
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        # 确保目录存在，递归创建
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path) # FAISS 对象会将索引文件（如 index.faiss 和 index.pkl）写入指定目录
        logger.info(f"向量索引已保存到：{self.index_save_path}")

    def load_index(self):
        """
        从磁盘加载已保存的FAISS索引，恢复self.vectorstore。如果路径不存在/加载失败，返回None
        :return:
        """

        # 首先确保 self.embeddings 可用；若为 None（可能未调用 setup_embeddings），则主动调用一次初始化嵌入模型。
        # 这是因为 FAISS.load_local 需要传入嵌入对象来重建向量检索能力。
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在：{self.index_save_path}。将构建新索引")
            return None


        try:
            # 第一个参数：索引文件所在目录。
            # 第二个参数：self.embeddings，用于恢复向量检索时的查询向量化功能。
            # allow_dangerous_deserialization=True：因为 FAISS 的本地文件可能包含序列化对象，该参数显式允许加载，避免安全警告（生产环境中应注意文件来源可信性）。
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization= True
            )
            logger.info(f"向量索引已从{self.index_save_path}加载")
            return self.vectorstore

        except Exception as e:
            logger.warning(f"加载向量索引失败：{e}，将构建新索引")
            return None

    def similarity_search(self, query:str, k:int = 5) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        # 直接委托给 self.vectorstore.similarity_search(query, k=k)：
        # 该方法内部会先用 self.embeddings 将 query 向量化，
        # 然后在 FAISS 索引中按余弦相似度（或其他度量）检索最相似的 k 个向量，并返回对应的 Document 列表。
        return self.vectorstore.similarity_search(query, k=k)