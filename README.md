# 尝尝咸淡 RAG 系统

基于 RAG（检索增强生成）的智能菜谱问答系统，解决"今天吃什么"的世纪难题。

## 功能特点

- **混合检索** — FAISS 向量检索 + BM25 关键词检索，通过 RRF 融合排序，兼顾语义和关键词匹配
- **智能查询路由** — 自动识别列表推荐、详细菜谱、通用问答三种意图
- **父子文档模式** — 细粒度 Markdown 标题分块用于检索，完整菜谱文档用于生成回答
- **元数据过滤** — 支持按菜品分类（荤菜、素菜、汤品等）和烹饪难度筛选
- **流式/非流式输出** — 支持两种回答生成模式

## 快速开始

### 环境要求

- Python 3.10+
- [阿里云百炼](https://bailian.console.aliyun.com/) API Key（DashScope）

### 安装依赖

```bash
pip install langchain langchain-community langchain-text-splitters langchain-core faiss-cpu dashscope python-dotenv
```

### 配置

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=你的阿里云百炼API密钥
```

### 运行

```bash
python main.py
```

首次运行会自动构建向量索引（处理 data/ 目录下的全部菜谱），后续运行直接加载已保存的索引。

## 项目结构

```
cook-rag-1/
├── main.py                 # 系统入口，交互式问答循环
├── config.py               # RAGConfig 配置类
├── rag_modules/            # 核心 RAG 管线模块
│   ├── data_preparation.py       # 数据加载、元数据增强、Markdown 分块
│   ├── index_construction.py     # FAISS 向量索引构建/加载
│   ├── retrieval_optimization.py # 混合检索 + RRF 重排序
│   └── generation_integration.py # 查询路由/重写 + LLM 回答生成
├── data/C8/cook/dishes/    # 323 道中文菜谱（Markdown）
│   ├── meat_dish/          # 荤菜
│   ├── vegetable_dish/     # 素菜
│   ├── aquatic/            # 水产
│   ├── soup/               # 汤品
│   ├── staple/             # 主食
│   ├── dessert/            # 甜品
│   ├── drink/              # 饮品
│   ├── breakfast/          # 早餐
│   ├── condiment/          # 调料
│   └── semi-finished/      # 半成品
└── vector_index/           # 持久化的 FAISS 索引文件
```

## 技术栈

| 组件 | 技术 |
|---|---|
| 嵌入模型 | 阿里云百炼 text-embedding-v2 |
| 大语言模型 | 通义千问 qwen-plus |
| 向量数据库 | FAISS (AVX2) |
| 关键词检索 | BM25 (LangChain) |
| 重排序 | Reciprocal Rank Fusion (RRF) |

## 查询示例

- "推荐几个简单的素菜" — 按分类和难度筛选的列表推荐
- "宫保鸡丁怎么做" — 分步骤的详细菜谱
- "有什么适合夏天的汤" — 通用推荐查询
- "难度：简单 分类：水产" — 元数据精确过滤

## 数据来源

菜谱数据来自 [Anduin2017/HowToCook](https://github.com/Anduin2017/HowToCook) 项目，遵循其原始许可证。
