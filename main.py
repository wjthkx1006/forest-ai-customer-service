"""
森林二手手机智能客服 - 优化版
专注于响应速度和性能优化
"""
import os
import time
import hashlib
import streamlit as st
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# 加载环境变量
load_dotenv()

# 导入工具模块
from utils.logger import logger
from utils.chat_utils import contains_sensitive_content, process_sensitive_content, process_special_questions, generate_unrelated_response
from utils.prompts import build_prompt_with_context, build_prompt_no_context, build_fallback_prompt
from utils.ai_service import get_ai_service
from utils.context_processor import get_context_processor

# 配置
# 优先从 Streamlit Secrets 读取，兼容本地 .env（方便部署）
DASHSCOPE_API_KEY = st.secrets.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
WECHAT_ID = os.getenv("WECHAT_ID", "123456789")
PHONE_NUMBER = os.getenv("PHONE_NUMBER", "123456789")

# 示例问题
SUGGESTED_QUESTIONS = [
    "怎么查手机序列号？",
    "二手机支持5G吗？",
    "有锁机和无锁机区别？",
    "买二手iPhone要注意什么？",
    "二手机可以退货吗？",
    "如何验机是否无拆无修？",
    "二手机电池健康多少合适？",
    "屏幕发黄正常吗？"
]


# ---------- 初始化向量数据库（优化版） ----------
_vector_db_instance = None

@st.cache_resource
def get_vector_db():
    """获取向量数据库实例（单例模式）"""
    global _vector_db_instance
    
    if _vector_db_instance is None:
        logger.info("初始化向量数据库...")
        try:
            embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=DASHSCOPE_API_KEY)
            _vector_db_instance = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings
            )
            logger.info("向量数据库初始化成功")
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")
            _vector_db_instance = None
    
    return _vector_db_instance


# ---------- 快速知识库检索 ----------
def fast_knowledge_retrieval(query: str, vector_db) -> str:
    """快速知识库检索"""
    if not vector_db:
        return ""
    
    try:
        # 使用更快的检索参数
        docs = vector_db.similarity_search(query, k=3)  # 减少检索数量
        if docs:
            # 只取前2个文档，减少处理时间
            knowledge = "\n".join([doc.page_content for doc in docs[:2]])
            logger.info(f"快速检索到 {len(docs)} 个相关文档")
            return knowledge
    except Exception as e:
        logger.error(f"快速知识库检索失败: {e}")
    
    return ""


# ---------- 智能回答函数（极速优化版） ----------
@st.cache_data(ttl=3600)
def smart_answer_fast(query: str, messages: List[Dict]) -> Dict[str, Any]:
    """极速优化版智能回答，专注于响应速度"""
    start_time = time.time()
    logger.info(f"处理用户问题（快速）: {query}")

    # 1. 敏感内容检测（快速检查）
    if contains_sensitive_content(query):
        result = process_sensitive_content(query)
        result["response_time"] = time.time() - start_time
        return result

    # 2. 特殊问题处理（快速响应）
    special_result = process_special_questions(query, messages)
    if special_result:
        special_result["response_time"] = time.time() - start_time
        return special_result

    # 3. 智能缓存检查（优化版）
    cache_key = generate_cache_key_fast(query, messages)
    if cache_key in st.session_state:
        result = st.session_state[cache_key]
        # 检查缓存是否过期（超过1小时）
        if time.time() - result.get("cache_timestamp", 0) < 3600:
            result["from_cache"] = True
            result["response_time"] = time.time() - start_time
            logger.info(f"从缓存获取回答（快速），缓存键: {cache_key[:20]}...")
            return result
        else:
            # 缓存过期，删除
            del st.session_state[cache_key]

    # 4. 快速上下文分析（简化版）
    context_processor = get_context_processor()
    context_analysis = context_processor.analyze_conversation_context(messages, query)
    
    # 5. 快速判断是否无关问题
    is_unrelated = is_unrelated_question_fast(query)
    
    # 如果是无关问题，使用快速响应
    if is_unrelated:
        response = generate_unrelated_response(query, context_analysis)
        result = build_result_fast(response, False, False, start_time)
        # 无关问题不缓存
        return result

    # 6. 并行处理：同时进行知识库检索和AI服务准备
    vector_db = get_vector_db()
    ai_service = get_ai_service()
    
    # 快速知识库检索
    knowledge = fast_knowledge_retrieval(query, vector_db)
    
    # 7. 快速回答生成
    if ai_service and hasattr(ai_service, 'client'):
        try:
            if knowledge:
    # 使用简化提示
                prompt = build_prompt_no_context(query, knowledge)
                response = ai_service.generate_answer(prompt, max_tokens=500)
            else:
                # 无知识库内容，使用AI服务直接回答
                prompt = f"请回答以下问题：{query}"
                response = ai_service.generate_answer(prompt, max_tokens=500)
        except Exception as e:
            logger.error(f"AI快速回答生成失败: {e}")
            response = generate_fallback_answer_fast(query)
    else:
        logger.warning("AI服务不可用，使用快速降级模式")
        response = generate_fallback_answer_fast(query)

    # 8. 构建结果
    result = build_result_fast(response, False, False, start_time)
    
    # 9. 快速缓存结果
    if "抱歉" not in response and "无法回答" not in response:
        cache_result_fast(cache_key, result)
    
    logger.info(f"快速回答生成完成，耗时: {result['response_time']:.2f}秒")
    return result


def generate_cache_key_fast(query: str, messages: List[Dict]) -> str:
    """生成快速缓存键"""
    # 简化缓存键生成
    recent_context = ""
    if len(messages) > 0:
        # 只取最后一条用户消息
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")[:30]  # 只取前30字符
                break
        
        if last_user_msg:
            recent_context = last_user_msg
    
    # 创建简化哈希键
    key_string = f"{query[:50]}_{recent_context}"  # 限制长度
    return f"fast_cache_{hashlib.md5(key_string.encode()).hexdigest()[:12]}"


def is_unrelated_question_fast(query: str) -> bool:
    """快速判断是否无关问题"""
    query_lower = query.lower()
    
    # 快速关键词匹配
    unrelated_keywords = [
        "天气", "美食", "餐厅", "电影", "电视剧", "娱乐", "明星",
        "体育", "运动", "比赛", "健身", "新闻", "时事", "热点",
        "你好", "嗨", "hello", "hi", "谢谢", "感谢"
    ]
    
    return any(keyword in query_lower for keyword in unrelated_keywords)


def generate_fallback_answer_fast(query: str) -> str:
    """快速降级回答"""
    return f"关于\"{query}\"，我目前没有足够的信息来详细回答。如果您有二手手机相关的问题，我很乐意为您解答。"


def build_result_fast(response: str, show_human: bool, is_sensitive: bool, start_time: float) -> Dict[str, Any]:
    """快速构建结果"""
    return {
        "response": response,
        "show_human": show_human,
        "is_sensitive": is_sensitive,
        "from_cache": False,
        "response_time": time.time() - start_time
    }


def cache_result_fast(cache_key: str, result: Dict[str, Any]):
    """快速缓存结果"""
    result["cache_timestamp"] = time.time()
    st.session_state[cache_key] = result
    logger.info(f"快速缓存回答，键: {cache_key[:20]}...")


# ---------- 主应用函数 ----------
def main_optimized():
    """优化版主应用"""
    st.set_page_config(
        page_title="森林二手手机智能客服",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # 自定义CSS
    st.markdown("""
    <style>
    /* 简化CSS，提高加载速度 */
    body {
        font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
        background-color: #f5f7fa;
    }
    
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .title {
        text-align: center;
        color: #1e88e5;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    
    .example-question-btn {
        background-color: #f0f7ff;
        border: 1px solid #1e88e5;
        color: #1e88e5;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .example-question-btn:hover {
        background-color: #1e88e5;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(30, 136, 229, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # 初始化session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 标题
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">📱 森林二手手机智能客服</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">专业解答二手手机相关问题，快速响应</p>', unsafe_allow_html=True)

    # 显示示例问题（始终显示）
    st.markdown("---")
    st.markdown("示例问题")
    st.markdown("点击以下问题快速开始：")
    
    # 创建两列布局显示示例问题
    cols = st.columns(2)
    
    # 初始化示例问题点击状态
    if "example_clicked" not in st.session_state:
        st.session_state.example_clicked = False
        st.session_state.clicked_question = None
    
    # 检查是否有示例问题被点击
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(question, key=f"example_{i}", use_container_width=True):
                # 当点击示例问题时，记录点击状态
                st.session_state.example_clicked = True
                st.session_state.clicked_question = question
                st.rerun()
    
    st.markdown("---")

    # 聊天历史
    st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
    
    # 显示所有历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 处理示例问题点击
    if st.session_state.example_clicked and st.session_state.clicked_question:
        clicked_question = st.session_state.clicked_question
        
        # 添加用户消息
        with st.chat_message("user"):
            st.markdown(clicked_question)
        st.session_state.messages.append({"role": "user", "content": clicked_question})
        
        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在快速思考..."):
                result = smart_answer_fast(clicked_question, st.session_state.messages)
                st.markdown(result["response"])
                
                # 显示转人工按钮
                if result.get("show_human") or result.get("is_sensitive"):
                    st.markdown("---")
                    st.error("抱歉，我无法回答这个问题")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.link_button("加微信咨询", f"weixin://dl/chat?{WECHAT_ID}", use_container_width=True)
                    with col2:
                        st.link_button("拨打电话", f"tel:{PHONE_NUMBER}", use_container_width=True)
        
        # 添加助手消息到历史
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
        
        # 重置点击状态
        st.session_state.example_clicked = False
        st.session_state.clicked_question = None
    
    st.markdown("</div>", unsafe_allow_html=True)

    # 处理聊天输入框输入
    prompt = st.chat_input("请输入您的问题...")
    if prompt:
        # 添加用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在快速思考..."):
                result = smart_answer_fast(prompt, st.session_state.messages)
                st.markdown(result["response"])
                
                # 显示转人工按钮
                if result.get("show_human") or result.get("is_sensitive"):
                    st.markdown("---")
                    st.error("抱歉，我无法回答这个问题")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.link_button("加微信咨询", f"weixin://dl/chat?{WECHAT_ID}", use_container_width=True)
                    with col2:
                        st.link_button("拨打电话", f"tel:{PHONE_NUMBER}", use_container_width=True)
        
        # 添加助手消息到历史
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main_optimized()
