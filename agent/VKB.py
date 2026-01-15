import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import hashlib

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"  
DB_PATH = os.path.join(os.path.dirname(__file__), 'database')
KB_FILE = os.path.join(os.path.dirname(__file__), 'knowledge_base.docx')

_collection = None
_reranker = None  

def _get_reranker():
    global _reranker
    if _reranker is None:
        print("正在加载重排序模型 (CrossEncoder)...")
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512) 
    return _reranker

def _compute_kb_fingerprint(kb_path: str) -> str:
    hasher = hashlib.md5()
    if not os.path.exists(kb_path):
        return ""
    with open(kb_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

from docx import Document  

def _read_file_content(file_path: str) -> str:
    if file_path.lower().endswith('.docx'):
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    elif file_path.lower().endswith('.txt'):
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

def _process_text_splitting(file_path):
    text = _read_file_content(file_path)  
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    print(f"原始文本已切分为 {len(chunks)} 个语义块")
    return chunks

def _initialize_kb_collection():
    global _collection
    os.makedirs(DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu" 
    )

    current_fingerprint = _compute_kb_fingerprint(KB_FILE)
    fingerprint_file = os.path.join(DB_PATH, "kb_fingerprint.txt")
    
    saved_fingerprint = None
    if os.path.exists(fingerprint_file):
        with open(fingerprint_file, 'r', encoding='utf-8') as f:
            saved_fingerprint = f.read().strip()

    if saved_fingerprint != current_fingerprint:
        print("检测到知识库变动，正在重建索引...")
        try:
            client.delete_collection("medical_kb")
        except Exception:
            pass

        _collection = client.create_collection(
            name="medical_kb",
            embedding_function=embedding_func
        )

        if os.path.exists(KB_FILE):
            docs = _process_text_splitting(KB_FILE)
            
            if docs:
                batch_size = 100
                total_batches = (len(docs) + batch_size - 1) // batch_size
                print(f"正在写入向量库，共 {len(docs)} 条数据...")
                
                for i in range(0, len(docs), batch_size):
                    batch_docs = docs[i : i + batch_size]
                    batch_ids = [f"id_{j}" for j in range(i, i + len(batch_docs))]
                    batch_metadatas = [{"index": j} for j in range(i, i + len(batch_docs))]
                    
                    _collection.add(
                        documents=batch_docs,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
            else:
                print("Warning: 知识库文件为空")
        
        with open(fingerprint_file, 'w', encoding='utf-8') as f:
            f.write(current_fingerprint)
            
    else:
        _collection = client.get_collection(
            name="medical_kb",
            embedding_function=embedding_func
        )
        print("加载现有知识库集合完成。")

def search_knowledge_base(query: str, top_k: int = 3):
    global _collection
    if _collection is None:
        _initialize_kb_collection()
    candidate_k = min(top_k * 5, 20) 
    results = _collection.query(
        query_texts=[query],
        n_results=candidate_k
    )
    documents = results['documents'][0]
    if not documents:
        return []
    reranker = _get_reranker()
    pairs = [[query, doc] for doc in documents]
    scored_results = reranker.predict(pairs)
    scored_results.sort(key=lambda x: x[1], reverse=True)
    final_results = [doc for doc, score in scored_results[:top_k]]
        
    return final_results

_initialize_kb_collection()