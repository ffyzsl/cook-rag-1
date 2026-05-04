"""
数据准备模块
"""
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import uuid

# 获取本模块的日志记录器，方便调试
logger = logging.getLogger(__name__)

class DataPreparationModule:
    """
    数据准备模块：负责数据加载、清洗和预处理
    """

    # ------定义分类和难度映射的类变量--------
    # 文件名是英文key，我们映射为中文标签，显示给用户。
    CATEGORY_MAPPING = {
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早餐",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品"
    }

    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = [
        "非常简单",
        "简单",
        "中等",
        "困难",
        "非常困难"
    ]

    def __init__(self, data_path: str):
        """
        初始化数据准备模块
        :param data_path: 数据文件路径
        """
        self.data_path = data_path                  # 保存菜谱 Markdown 文件所在目录的路径
        self.documents: List[Document] = []         # 存放原始完整菜谱文档（未分割的父文档），每个document对应一个 .md 文件。
        self.chunks: List[Document] = []            # 存放按 Markdown 标题分割后的子块，用于构建向量索引和检索。存放所有父文档切分后产生的子文档的集合。
        self.parent_child_map: Dict[str,str] = {}   # 字典映射，key为子块的 chunk_id，value为父文档的 parent_id（即文件级别的唯一ID）。这样当检索到某个子块时，能快速找回它所属的完整菜谱，以便生成回答时提供完整上下文。


    def load_documents(self) -> List[Document]:
        """
        加载文档数据，即将所有的md文件加载成为document
        :return: 加载的文档列表
        """
        logger.info(f"正在从{self.data_path}加载文档...")

        # 直接读取Markdown文件以保持原始格式
        documents = []
        data_path_obj = Path(self.data_path)

        # rglob 是 pathlib.Path 的方法，递归遍历目录下所有匹配 *.md 的文件。无论子目录多深都能找到。
        for md_file in data_path_obj.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read() # 返回整个文本的文本内容，str类型
                    try:
                        # Path(self.data_path): 将类属性 self.data_path转换为 Path 对象。
                        # .resolve(): 核心操作。它的作用是获取该路径的绝对路径，并且会解析路径中所有的符号链接（symlinks），以及消除路径中的 ..（上一级）和 .（当前级）。
                        # data_root 变量现在存储的是一个代表数据根目录的、最严格的绝对路径对象。
                        data_root = Path(md_file.data_path).resolve()

                        # Path(md_file).resolve(): 同样地，把目标 Markdown 文件路径变成绝对路径。
                        # .relative_to(data_root): 尝试计算相对路径
                        # 功能为：如果 md 文件是 D:/project/data/notes/test.md，而 data_root 是 D:/project/data，那么这一步会返回 Path('notes/test.md')
                        relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()

                        # .as_posix(): 标准化路径分隔符。在 Web 开发、Markdown 图片/相对链接中，标准要求使用 /。
                    except Exception:

                        # 如果无法计算出相对路径，代码放弃了寻找相对关系，直接把原始传入的 md_file 变成 Path 对象，然后转换成带正斜杠 / 的 POSIX 格式字符串。
                        relative_path = Path(md_file).as_posix()

                    # 用 MD5 对相对路径字符串编码后的字节进行哈希，得到一个十六进制字符串（32个字符）。这保证了相同文件每次得到的 ID 相同，不同文件 ID 几乎不可能冲突。
                    parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                    doc = Document(
                        page_content= content,
                        metadata={
                            "source": str(md_file),
                            "parent_id": parent_id,
                            "doc_type": "parent"

                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")

        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")

        return documents

    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据，用来补充分类、菜品名称和难度等信息
        :param self:
        :param doc: 需要增强元数据的文档
        :return:
        """

        # 从元数据中取出源文件路径，转为 Path 对象，然后拆分为路径各部分元组，如 ('cook', 'meat_dish', '红烧肉.md')。
        file_path = Path(doc.metadata.get("source",''))
        # .parts	路径各组成部分的元组
        # 例如文件路径 data/cook/meat_dish/红烧肉.md，path_parts 会是 ('data', 'cook', 'meat_dish', '红烧肉.md')
        path_parts = file_path.parts

        # 提取菜品分类
        doc.metadata["category"] = "其他" # 默认分类为其他

        # 遍历CATEGORY_MAPPING，检查路径中是否包含英文目录名，如果包含，则设置分类为对应的中文名称（如 '荤菜'），找到即跳出循环。
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata["category"] = value
                break

        # 提取菜品名称
        # 菜品名称直接取文件名（不含后缀），例如 红烧肉.md → '红烧肉'。这种约定要求文件名就是菜谱名称。
        # .stem	不带扩展名的文件名
        doc.metadata["dish_name"] = file_path.stem

        # 分析难度等级
        # 通过字符串 '★' 的数量来判断难度。这是与数据格式的约定，菜谱正文中会用实心五角星个数表示难度。
        # 注意 elif 顺序：从多到少匹配，避免子串误判（比如 '★★★' 包含 '★'，但先匹配多星，所以安全）
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

        # 最终每个文档的 metadata 会包含 source、parent_id、doc_type、category、dish_name、difficulty 等丰富信息。

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """
        对外提供支持的分类标签列表
        :param cls:
        :return: 列表
        """
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """
        对外提供支持的难度标签列表
        :return:
        """
        return cls.DIFFICULTY_LABELS

    def chunk_documents(self) -> List[Document]:
        """
        Markdown结构感知分块，将一个md文件切成若干了小块
        :return: 分块后的文档列表
        """
        logger.info("正在进行Markdown结构感知分块……")

        if not self.documents:
            raise ValueError("请先加载文档")

        # 调用内部方法执行具体的分块逻辑,得到分块列表
        chunks = self._markdown_header_split()

        # 为每个chunks添加基础元数据
        for i, chunk in enumerate(chunks):
            # 如果分割器没有自动生成ID，则手动生成一个UUID
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["batch_index"] = i # 当前批次中的顺序索引，便于调试。
            chunk.metadata["chunk_size"] = len(chunk.page_content) # 块的内容长度，用于统计。

        self.chunks = chunks # 最后把结果存入 self.chunks。
        logger.info(f"Markdown分块完成，共生成{len(chunks)}个chunks")
        return chunks


    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割
        :return: 按标题结构分割文档列表
        """

        # 遇到 # ## ### 这三种级别的标题时就会切开文档，每个块会带上它所属的标题层级信息（作为元数据）。
        headers_to_split_on = [
            ("#","主标题"),
            ("##","二级标题"),
            ("###","三级标题")
        ]

        # MarkdownHeaderTextSplitter: 定义你想要根据哪些等级的标题进行切分，并为这些标题指定在元数据中的key
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题，便于理解上下文
        )

        all_chunks = []
        # 遍历所有父文档
        for doc in self.documents:
            try:
                content_preview = doc.page_content[:200] # 取文档内容的前 200 个字符作为预览，用于后续检查是否包含标题。

                # 将内容按行分割，检查每一行以去除首尾空白后是否以 # 开头（即是否为标题行），若任何一行满足条件，has_headers 为 True。
                has_header = any(line.strip().startswith("#") for line in content_preview.split("\n"))

                if not has_header:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 内容中没有发现Markdown标题")
                    logger.debug(f"内容预览: {content_preview}")

                # 调用分割器的 split_text 方法，将文档的完整内容 doc.page_content 按标题层级切分成多个 Document 对象。
                # 返回的列表 md_chunks 包含所有子块，每个子块已经带有 langchain 自动添加的元数据（如标题层级信息）。
                md_chunks = markdown_splitter.split_text(doc.page_content)

                logger.debug(f"文档 {doc.metadata.get('dish_name', '未知')} 分割成 {len(md_chunks)} 个chunk")

                if len(md_chunks) <= 1:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 未能按标题分割，可能缺少标题结构")

                # 获取父文档的id
                parent_id = doc.metadata["parent_id"]

                # 遍历该md切出的chunks来添加额外信息
                for i,chunk in enumerate(md_chunks):
                    # 使用 uuid.uuid4() 生成一个全局唯一的字符串标识，作为子块的 ID。
                    child_id = str(uuid.uuid4())

                    # # 合并原文档元数据和新的标题元数据
                    chunk.metadata.update(doc.metadata) # 首先用父文档的元数据更新子块的元数据，使其继承分类、菜名、难度等信息。
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",
                        "chunk_index": i,
                    })
                    # 将父子映射关系填入映射表中
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                all_chunks.append(doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        按分类过滤文档

        Args:
            category: 菜品分类

        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('category') == category]

    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        按难度过滤文档

        Args:
            difficulty: 难度等级

        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('difficulty') == difficulty]

    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}

        categories = {}
        difficulties = {}

        for doc in self.documents:
            category = doc.metadata.get("category", '未知')
            categories[category] = categories.get(category, 0) + 1
            difficulty = doc.metadata.get("difficulty",'未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'difficulties': difficulties,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }


    def export_metadata(self, output_path: str):
        """
         将文档的关键元数据保存为 JSON，便于离线分析或检查。
        :param output_path:
        :return:
        """
        import json
        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据检索到的子块找出对应的完整父文档，去重并按照相关性排序后返回
        :param child_chunks: 检索到的子块
        :return:
        """
        parent_relevance = {} # 记录每个父文档被命中的子块数量
        parent_docs_map = {}  # 缓存父文档对象# ，key为parent_id，避免后续重复遍历self.documents

        # 遍历检索到的每一个子块
        for chunk in child_chunks:
            parent_id = chunk.metadata.get('parent_id') # 从子块的元数据中取出parent_id
            if parent_id: # 当前子块有parent_id，跳过未正确关联的块

                # 累加该父文档命中次数：若 parent_id 已存在，取出当前计数 +1；不存在则默认 0，再 +1。这一步建立了父文档的“相关性评分”。
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 如果 parent_id 尚未缓存，则去 self.documents 中查找并缓存
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get('parent_id') == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按照相关性降序排序分文档ID
        # parent_relevance.keys() 返回所有被命中的父文档 parent_id 集合。
        # sorted 函数基于 key=lambda x: parent_relevance[x] 对 parent_id 排序。lambda 表达式将每个 parent_id 映射为其对应的命中计数（相关性分数）。
        # reverse=True 表示降序排列，即命中次数最多的父文档排在最前面。
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                   key=lambda x: parent_relevance[x],
                                   reverse=True)

        # 初始化一个空列表用来存放最终结果
        parent_docs = []
        # 遍历排序后的parent_id列表，从parent_docs_map中取出对应的document对象追加到结果列表中
        for parent_id in sorted_parent_ids:
            parent_docs.append(parent_docs_map[parent_id])


        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs










