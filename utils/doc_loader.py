"""
文档加载器 - 读取 docs/ 目录下的文档并进行文本切片
支持 .txt、.md、.pdf 格式
"""
import os
import re
from typing import List, Tuple
from .logger import logger

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(docs_dir: str = "./docs") -> List[str]:
    """读取 docs/ 目录下所有文档并切片，返回文本块列表"""
    if not os.path.exists(docs_dir):
        logger.warning(f"文档目录不存在: {docs_dir}")
        return []

    all_chunks: List[str] = []
    for filename in sorted(os.listdir(docs_dir)):
        filepath = os.path.join(docs_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        if not os.path.isfile(filepath):
            continue

        try:
            if ext == ".pdf":
                text = _read_pdf(filepath)
            else:
                text = _read_text(filepath)

            if not text or not text.strip():
                continue

            chunks = _split_text(text, filename)
            all_chunks.extend(chunks)
            logger.info(f"文档 {filename} 切片完成，共 {len(chunks)} 个文本块")
        except Exception as e:
            logger.error(f"加载文档 {filename} 失败: {e}")

    logger.info(f"文档加载完成，共 {len(all_chunks)} 个文本块")
    return all_chunks


def _read_text(filepath: str) -> str:
    """读取 txt/md 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf(filepath: str) -> str:
    """读取 PDF 文件"""
    try:
        from pypdf import PdfReader

        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except ImportError:
        logger.error("pypdf 未安装，无法读取 PDF 文件")
        return ""


def _split_text(text: str, source_filename: str = "") -> List[str]:
    """递归字符切分，优先保留语义边界"""
    separators = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]
    chunks = _recursive_split(text, separators, CHUNK_SIZE, CHUNK_OVERLAP)

    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if source_filename:
            result.append(f"[来源：{source_filename}]\n{chunk}")
        else:
            result.append(chunk)
    return result


def _recursive_split(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """递归文本切分"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = ""
    for sep in separators:
        if sep in text:
            separator = sep
            break

    if not separator:
        return _hard_split(text, chunk_size, chunk_overlap)

    parts = text.split(separator)
    chunks = []
    current = ""

    for part in parts:
        if not part.strip():
            continue

        candidate = current + separator + part if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(part) > chunk_size:
                sub_chunks = _recursive_split(part, separators[1:] if len(separators) > 1 else [], chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current)

    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, chunk_overlap)

    return chunks


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """硬切分：按固定字符数切分"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - chunk_overlap
    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """为切片添加前后重叠"""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        result.append(prev_tail + chunks[i])
    return result


def load_faq(faq_path: str = "faq.json") -> List[str]:
    """加载 FAQ 知识库"""
    import json

    if not os.path.exists(faq_path):
        logger.warning(f"FAQ文件不存在: {faq_path}")
        return []

    try:
        with open(faq_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [f"问题：{item['question']}\n答案：{item['answer']}" for item in data]
        logger.info(f"FAQ加载完成，共 {len(texts)} 条")
        return texts
    except Exception as e:
        logger.error(f"加载FAQ失败: {e}")
        return []


def load_all_knowledge(
    faq_path: str = "faq.json",
    docs_dir: str = "./docs"
) -> Tuple[List[str], List[str]]:
    """加载所有知识源：FAQ + 文档，分别返回"""
    faq_texts = load_faq(faq_path)
    doc_chunks = load_documents(docs_dir)
    return faq_texts, doc_chunks
