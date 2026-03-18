"""
构建向量数据库 - 将FAQ知识库转换为向量存储
"""
import json
from typing import List

import requests
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from config import DASHSCOPE_API_KEY


class DashScopeEmbeddings(Embeddings):
    """阿里云DashScope文本嵌入"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            payload = {"model": "text-embedding-v2", "input": batch_texts}
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            batch_embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": "text-embedding-v2", "input": [text]}
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["data"][0]["embedding"]


def build_vector_db():
    """构建向量数据库"""
    # 读取FAQ
    with open('faq.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将问答对拼接成文本块
    texts = [f"问题：{item['question']}\n答案：{item['answer']}" for item in data]

    # 初始化嵌入模型
    embeddings = DashScopeEmbeddings(api_key=DASHSCOPE_API_KEY)

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
