"""
配置文件 - 集中管理应用参数（优化版）
本地开发：创建.env文件设置环境变量
部署环境：在Streamlit Cloud Secrets中设置
"""
import os

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==================== API配置 ====================
# 必须从环境变量读取，请勿在代码中硬编码
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# ==================== 性能优化配置 ====================
CACHE_MAX_SIZE = 100          # 缓存最大条目数
CACHE_TTL = 3600              # 缓存过期时间（秒）
SIMILARITY_THRESHOLD = 0.6    # 相似度阈值（降低以提高召回率）
MAX_CACHE_AGE = 3600          # 缓存最大年龄（秒）
MAX_CONVERSATION_HISTORY = 20 # 最大对话历史条数

# ==================== 向量数据库配置 ====================
VECTOR_STORE_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-v2"
MAX_RETRIEVAL_DOCS = 5        # 最大检索文档数
MIN_SIMILARITY_SCORE = 0.6    # 最小相似度分数

# ==================== AI模型配置 ====================
LLM_MODEL = "qwen-turbo"      # 使用的AI模型
LLM_TEMPERATURE = 0.3         # 温度参数（0-1，越低越确定）
LLM_MAX_TOKENS = 1000         # 最大生成token数
LLM_TIMEOUT = 30              # API调用超时时间（秒）

# ==================== 回答质量配置 ====================
MIN_ANSWER_QUALITY_SCORE = 50 # 最小回答质量分数
MAX_ANSWER_LENGTH = 500       # 最大回答长度（字符）
MIN_ANSWER_LENGTH = 20        # 最小回答长度（字符）

# ==================== 应用配置 ====================
APP_TITLE = "森林二手手机智能客服"
APP_LAYOUT = "centered"
APP_DESCRIPTION = "专业二手手机智能客服助手，为您解答购买、使用、验机等问题"
APP_VERSION = "2.0.0"

# ==================== 业务配置 ====================
BUSINESS_NAME = "森林二手手机店"
SUPPORT_PHONE = os.getenv("PHONE_NUMBER", "12345678901")
SUPPORT_WECHAT = os.getenv("WECHAT_ID", "123456789")
BUSINESS_HOURS = "9:00-21:00"

# ==================== 功能开关 ====================
ENABLE_AI_SERVICE = True      # 是否启用AI服务
ENABLE_CACHE = True           # 是否启用缓存
ENABLE_STREAMING = False      # 是否启用流式响应（当前版本禁用）
ENABLE_QUALITY_CHECK = True   # 是否启用回答质量检查
