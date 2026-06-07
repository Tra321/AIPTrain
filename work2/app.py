import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List

try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import BaseCrossEncoder
except ModuleNotFoundError:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain.retrievers.document_compressors.cross_encoder_rerank import BaseCrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "软件与人工智能学院本科生学业预警实施办法.pdf"
DEFAULT_DB_DIR = BASE_DIR / "chroma_db"
DEFAULT_MODEL_CACHE_DIR = BASE_DIR / "models"
COLLECTION_NAME = "academic_warning_policy"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 LangChain 的学业预警 RAG 知识问答系统"
    )
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="唯一知识来源 PDF 路径")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="Chroma 向量库目录")
    parser.add_argument(
        "--model-cache-dir",
        default=str(DEFAULT_MODEL_CACHE_DIR),
        help="Embedding 和 Reranker 的本地缓存目录",
    )
    parser.add_argument("--model", default="qwen3.5:0.8b", help="Ollama 本地大语言模型名称")
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-zh-v1.5",
        help="HuggingFace Embedding 模型",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-base",
        help="Cross-Encoder 重排序模型",
    )
    parser.add_argument("--chunk-size", type=int, default=600, help="文本块大小")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="文本块重叠长度")
    parser.add_argument("--vector-k", type=int, default=12, help="向量检索召回数量")
    parser.add_argument("--bm25-k", type=int, default=12, help="BM25 检索召回数量")
    parser.add_argument("--rerank-top-k", type=int, default=6, help="重排序后保留数量")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度")
    parser.add_argument("--rebuild", action="store_true", help="强制重建 Chroma 向量库")
    parser.add_argument("--offline", action="store_true", help="只读取本地已下载模型，不联网下载")
    parser.add_argument("--debug", action="store_true", help="输出检索到的参考片段，便于排查回答质量")
    parser.add_argument("--web", action="store_true", help="启动 Streamlit Web UI")
    return parser


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    """中文按字召回，英文和数字按词召回，提升 BM25 对中文问题的命中率。"""
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
    bigrams = [
        tokens[index] + tokens[index + 1]
        for index in range(len(tokens) - 1)
        if re.match(r"^[\u4e00-\u9fff]{2}$", tokens[index] + tokens[index + 1])
    ]
    return tokens + bigrams


def format_document_for_context(doc: Document) -> str:
    page = doc.metadata.get("page", "未知")
    chunk_id = doc.metadata.get("chunk_id", "未知")
    return f"来源：第 {page} 页，片段 {chunk_id}\n内容：{doc.page_content}"


def load_and_split_pdf(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> List[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到 PDF 文件：{pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    docs = splitter.split_documents(pages)

    for index, doc in enumerate(docs):
        doc.page_content = normalize_text(doc.page_content)
        doc.metadata["chunk_id"] = index
        doc.metadata["source"] = pdf_path.name
        if "page" in doc.metadata:
            doc.metadata["page"] = int(doc.metadata["page"]) + 1

    return [doc for doc in docs if doc.page_content]


def build_embeddings(model_name: str, cache_dir: Path, offline: bool) -> HuggingFaceEmbeddings:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=str(cache_dir),
        model_kwargs={"device": "cpu", "local_files_only": offline},
        encode_kwargs={"normalize_embeddings": True},
    )


class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    """把 sentence_transformers.CrossEncoder 包装成 LangChain 的 BaseCrossEncoder。"""

    def __init__(self, model_name: str, cache_dir: Path, offline: bool) -> None:
        self._client = CrossEncoder(
            model_name_or_path=model_name,
            cache_dir=str(cache_dir),
            local_files_only=offline,
        )

    def score(self, text_pairs: List[List[str]]) -> List[float]:
        scores = self._client.predict(text_pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return [float(score) for score in scores]


def build_vector_store(
    docs: List[Document],
    embeddings: HuggingFaceEmbeddings,
    db_dir: Path,
    rebuild: bool,
) -> Chroma:
    if rebuild and db_dir.exists():
        import shutil

        shutil.rmtree(db_dir)

    if db_dir.exists() and any(db_dir.iterdir()):
        return Chroma(
            persist_directory=str(db_dir),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(db_dir),
        collection_name=COLLECTION_NAME,
    )


def format_sources(docs: Iterable[Document]) -> str:
    seen = set()
    lines = []
    for doc in docs:
        page = doc.metadata.get("page", "未知")
        chunk_id = doc.metadata.get("chunk_id", "未知")
        key = (page, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {doc.metadata.get('source', DEFAULT_PDF.name)} 第 {page} 页，片段 {chunk_id}")
    return "\n".join(lines) if lines else "- 未检索到可引用片段"


def create_rag_chain(args: argparse.Namespace):
    pdf_path = Path(args.pdf).resolve()
    db_dir = Path(args.db_dir).resolve()
    model_cache_dir = Path(args.model_cache_dir).resolve()

    docs = load_and_split_pdf(pdf_path, args.chunk_size, args.chunk_overlap)
    embeddings = build_embeddings(args.embedding_model, model_cache_dir, args.offline)
    vector_store = build_vector_store(docs, embeddings, db_dir, args.rebuild)

    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": args.vector_k},
    )

    bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=tokenize_for_bm25)
    bm25_retriever.k = args.bm25_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.65, 0.35],
    )

    reranker_model = SentenceTransformerCrossEncoder(
        model_name=args.reranker_model,
        cache_dir=model_cache_dir,
        offline=args.offline,
    )
    reranker = CrossEncoderReranker(model=reranker_model, top_n=args.rerank_top_k)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=reranker,
    )

    llm = ChatOllama(
        model=args.model,
        temperature=args.temperature,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
你是“软件与人工智能学院本科生学业预警实施办法”问答助手。
你必须严格遵守以下规则：
1. 只能依据<context>中的资料回答，不能使用常识、猜测或外部资料补充。
2. 如果资料中没有明确答案，直接回答“根据《软件与人工智能学院本科生学业预警实施办法》，未检索到相关明确规定。”
3. 回答要准确、简洁，必须引用依据所在页码。
4. 不要编造条款、数字、流程、机构或责任人。
5. 不要输出思考过程，不要输出 <think> 标签。

<context>
{context}
</context>
""".strip(),
            ),
            ("human", "/no_think\n{input}"),
        ]
    )

    document_prompt = PromptTemplate.from_template("{page_content}")
    qa_chain = create_stuff_documents_chain(
        llm,
        prompt,
        document_prompt=document_prompt,
        document_separator="\n\n---\n\n",
    )
    return {
        "retriever": compression_retriever,
        "qa_chain": qa_chain,
    }


def answer_question(chain, question: str) -> dict:
    context_docs = chain["retriever"].invoke(question)
    context_with_sources = [
        Document(
            page_content=format_document_for_context(doc),
            metadata=doc.metadata,
        )
        for doc in context_docs
    ]
    answer = chain["qa_chain"].invoke(
        {
            "input": question,
            "context": context_with_sources,
        }
    ).strip()
    return {
        "answer": answer,
        "sources": format_sources(context_docs),
        "documents": context_docs,
    }


def run_cli(args: argparse.Namespace) -> None:
    chain = create_rag_chain(args)
    print("学业预警 RAG 问答系统已启动。输入 exit / quit 退出。")
    while True:
        question = input("\n请输入问题：").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        result = answer_question(chain, question)
        print("\n回答：")
        print(result["answer"])
        print("\n参考来源：")
        print(result["sources"])
        if args.debug:
            print("\n检索片段：")
            for index, doc in enumerate(result["documents"], start=1):
                page = doc.metadata.get("page", "未知")
                chunk_id = doc.metadata.get("chunk_id", "未知")
                print(f"\n[{index}] 第 {page} 页，片段 {chunk_id}")
                print(doc.page_content[:800])


def run_web(args: argparse.Namespace) -> None:
    import streamlit as st

    st.set_page_config(page_title="学业预警 RAG 问答系统", layout="wide")
    st.title("学业预警知识问答系统")
    st.caption("唯一知识来源：《软件与人工智能学院本科生学业预警实施办法.pdf》")

    @st.cache_resource(show_spinner="正在加载 PDF、构建检索器和重排序模型...")
    def cached_chain(config_items):
        parser = build_arg_parser()
        cached_args = parser.parse_args([])
        for key, value in config_items:
            setattr(cached_args, key, value)
        return create_rag_chain(cached_args)

    config_items = tuple(sorted(vars(args).items()))
    chain = cached_chain(config_items)

    question = st.chat_input("请输入关于学业预警的问题")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索、重排序并生成答案..."):
                result = answer_question(chain, question)
            st.markdown(result["answer"])
            with st.expander("参考来源"):
                st.markdown(result["sources"])
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.web or os.environ.get("STREAMLIT_SERVER_PORT"):
        run_web(args)
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
