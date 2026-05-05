"""
构建向量数据库 - 将FAQ + 文档知识库转换为向量存储
"""
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

from config import DASHSCOPE_API_KEY
from utils.doc_loader import load_all_knowledge


def build_vector_db():
    """构建向量数据库（FAQ + 文档融合）"""
    faq_texts, doc_chunks = load_all_knowledge()
    all_texts = faq_texts + doc_chunks

    if not all_texts:
        print("没有找到任何知识内容，请检查 faq.json 和 docs/ 目录")
        return

    embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=DASHSCOPE_API_KEY)

    vectorstore = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"向量库构建完成！共导入 {len(all_texts)} 条知识（FAQ: {len(faq_texts)}, 文档: {len(doc_chunks)}）")


if __name__ == "__main__":
    build_vector_db()
