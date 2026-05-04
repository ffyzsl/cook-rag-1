from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

loader = TextLoader("../data/C8/cook/dishes/aquatic/小龙虾/小龙虾.md",encoding="utf-8")

docs = loader.load()

print(docs)

header_to_split_on = [
    ("#", "主标题"),
    ("##", "二级标题"),
    ("###", "三级标题")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=header_to_split_on,
    strip_headers=True, # 默认为True，表示在正文中去掉匹配到的标题符号
)

md_header_splits = markdown_splitter.split_text(docs[0].page_content)

print(md_header_splits)

for i, split in enumerate(md_header_splits):
    print(f"---文本块{i+1}---")
    print(f"内容：{split.page_content}")
    print(f"元数据:{split.metadata}\n")