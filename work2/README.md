# 学业预警 RAG 知识问答系统

本项目基于 LangChain 实现，唯一知识来源为：

`软件与人工智能学院本科生学业预警实施办法.pdf`

系统能力：

- `PyPDFLoader` 加载 PDF
- `RecursiveCharacterTextSplitter` 智能切分文档
- HuggingFace Embedding + Chroma 构建向量检索器
- BM25 构建关键词检索器
- `EnsembleRetriever` 混合检索，默认权重为向量 0.65、BM25 0.35
- `BAAI/bge-reranker-base` Cross-Encoder 精排
- Ollama + `qwen3.5:0.8b` 本地大模型生成答案
- 严格提示词约束，只能依据 PDF 内容回答
- 支持命令行交互和 Streamlit Web UI

## 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Ollama，并确保本地已有 `qwen3.5:0.8b` 模型：

```bash
ollama pull qwen3.5:0.8b
ollama serve
```

## 本地模型缓存

系统会把 HuggingFace 的 Embedding 模型和 Reranker 模型缓存到项目目录下的 `models/`：

- `BAAI/bge-small-zh-v1.5`
- `BAAI/bge-reranker-base`

第一次运行需要联网下载：

```bash
python app.py
```

下载完成后，可以使用离线模式，只读取本地缓存：

```bash
python app.py --offline
```

如果想换缓存目录：

```bash
python app.py --model-cache-dir D:\models\rag
```

Ollama 的 `qwen3.5:0.8b` 由 Ollama 自己管理，执行过 `ollama pull qwen3.5:0.8b` 后就是本地模型。

## 命令行问答

```bash
python app.py
```

首次运行会加载 PDF、下载 Embedding 与重排序模型，并在 `chroma_db/` 中生成向量库。第二次运行会复用 `models/` 和 `chroma_db/`，启动会快很多。

## Web UI

```bash
streamlit run app.py -- --web
```

## 常用参数

```bash
python app.py --rebuild
python app.py --model qwen3.5:0.8b
python app.py --offline
python app.py --debug
python app.py --model-cache-dir D:\models\rag
python app.py --vector-k 12 --bm25-k 12 --rerank-top-k 6
python app.py --embedding-model BAAI/bge-small-zh-v1.5
python app.py --reranker-model BAAI/bge-reranker-base
```

## 回答不准确时的排查

建议先重建一次向量库，避免旧索引影响结果：

```bash
python app.py --rebuild --offline
```

然后使用调试模式提问：

```bash
python app.py --offline --debug
```

调试模式会在答案后打印重排序后的原文片段。如果片段里没有问题对应的规定，说明检索参数需要调大；如果片段正确但答案仍错误，通常是本地生成模型太小，可以尝试换更大的 Ollama 模型。

## 回答边界

提示词要求模型只能使用检索到的 PDF 片段回答。若参考资料中没有明确规定，系统会回答：

`根据《软件与人工智能学院本科生学业预警实施办法》，未检索到相关明确规定。`
