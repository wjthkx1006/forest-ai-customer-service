"""
构建向量数据库 - 将FAQ知识库转换为向量存储
"""
import json
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

from config import DASHSCOPE_API_KEY


def build_vector_db():
    """构建向量数据库"""
    # 读取FAQ
    with open('faq.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将问答对拼接成文本块
    texts = [f"问题：{item['question']}\n答案：{item['answer']}" for item in data]

    # 初始化嵌入模型（与main.py保持一致）
    embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=DASHSCOPE_API_KEY)

    # 存入Chroma
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()
    print(f" 向量库构建完成！共导入 {len(texts)} 条知识")


if __name__ == "__main__":
    build_vector_db()
