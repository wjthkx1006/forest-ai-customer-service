"""
AI模型服务模块 - 提供智能回答生成功能
"""
import time
import json
from typing import List, Dict, Any, Optional
import dashscope
from dashscope import Generation
from openai import OpenAI
from .logger import logger


class AIService:
    """AI服务类，提供智能回答生成功能"""
    
    def __init__(self, api_key: str, model: str = None, temperature: float = None):
        """
        初始化AI服务
        
        Args:
            api_key: DashScope API密钥
            model: 模型名称
            temperature: 温度参数，控制回答的随机性
        """
        self.api_key = api_key
        self.model = model or "qwen-turbo"
        self.temperature = temperature or 0.3
        self.client = None
        self.timeout = 30  # API调用超时时间
        
        # 从配置读取参数
        try:
            from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
            self.model = LLM_MODEL
            self.temperature = LLM_TEMPERATURE
            self.max_tokens = LLM_MAX_TOKENS
            self.timeout = LLM_TIMEOUT
        except ImportError:
            self.max_tokens = 1000
        
        # 初始化客户端
        self._init_client()
    
    def _init_client(self):
        """初始化客户端"""
        try:
            # 使用DashScope的兼容模式
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            logger.info(f"AI服务初始化成功，使用模型: {self.model}")
        except Exception as e:
            logger.error(f"AI服务初始化失败: {e}")
            self.client = None
    
    def generate_answer(self, prompt: str, max_tokens: int = None, stream: bool = False) -> str:
        """
        生成回答
        
        Args:
            prompt: 提示词
            max_tokens: 最大token数（默认使用配置值）
            stream: 是否使用流式响应
            
        Returns:
            生成的回答文本
        """
        if not self.client:
            logger.error("AI客户端未初始化")
            return "抱歉，AI服务暂时不可用。"
        
        try:
            start_time = time.time()
            
            # 使用配置的max_tokens或传入的值
            tokens_to_use = max_tokens or self.max_tokens
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的二手手机客服助手。请直接回答问题，不要添加自我介绍或刻意的开场白。回答要自然、简洁、专业。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=tokens_to_use,
                stream=stream,
                timeout=self.timeout
            )
            
            if stream:
                # 流式响应处理
                collected_chunks = []
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        collected_chunks.append(chunk.choices[0].delta.content)
                answer = "".join(collected_chunks)
            else:
                answer = response.choices[0].message.content
            
            response_time = time.time() - start_time
            
            # 记录token使用情况
            if hasattr(response, 'usage') and response.usage:
                token_info = f"，token使用: {response.usage.total_tokens}"
            else:
                token_info = ""
            
            logger.info(f"AI回答生成完成，耗时: {response_time:.2f}秒{token_info}")
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"AI回答生成失败: {e}")
            return "抱歉，生成回答时出现错误。"
    
    def generate_answer_with_context(self, query: str, context: str, knowledge: str) -> str:
        """
        基于上下文和知识库生成回答
        
        Args:
            query: 用户问题
            context: 对话上下文
            knowledge: 知识库内容
            
        Returns:
            生成的回答
        """
        prompt = self._build_context_prompt(query, context, knowledge)
        return self.generate_answer(prompt)
    
    def generate_answer_without_context(self, query: str, knowledge: str) -> str:
        """
        基于知识库生成回答（无上下文）
        
        Args:
            query: 用户问题
            knowledge: 知识库内容
            
        Returns:
            生成的回答
        """
        prompt = self._build_simple_prompt(query, knowledge)
        return self.generate_answer(prompt)
    
    def _build_context_prompt(self, query: str, context: str, knowledge: str) -> str:
        """构建带上下文的提示词"""
        return f"""
        你是"森林二手手机智能客服"的AI助手，名字叫"森林"。
        你是一个专业、友好、耐心的客服，专门解答二手手机相关问题。

        【对话历史】
        {context if context else "这是第一次对话，没有历史记录。"}

        【相关知识库内容】
        {knowledge if knowledge else "知识库中没有找到相关信息。"}

        【当前用户问题】
        {query}

        请根据以上信息回答用户问题，要求：
        1. 如果对话历史中有相关信息，请结合历史对话进行回答
        2. 严格基于知识库内容回答，不要编造不存在的信息
        3. 如果知识库中没有相关信息，请诚实告知用户
        4. 回答要专业、准确、友好，使用自然的口语化表达
        5. 保持回答简洁明了，重点突出
        6. 如果用户的问题不清晰，可以适当追问或澄清
        7. 使用中文回答，语气亲切自然

        请开始你的回答：
        """
    
    def _build_simple_prompt(self, query: str, knowledge: str) -> str:
        """构建简单提示词（无上下文）"""
        return f"""
        你是"森林二手手机智能客服"的AI助手，名字叫"森林"。
        你是一个专业、友好、耐心的客服，专门解答二手手机相关问题。

        【相关知识库内容】
        {knowledge if knowledge else "知识库中没有找到相关信息。"}

        【用户问题】
        {query}

        请根据知识库内容回答用户问题，要求：
        1. 严格基于知识库内容回答，不要编造不存在的信息
        2. 如果知识库中没有相关信息，请诚实告知用户
        3. 回答要专业、准确、友好，使用自然的口语化表达
        4. 保持回答简洁明了，重点突出
        5. 使用中文回答，语气亲切自然

        请开始你的回答：
        """
    
    def validate_answer_quality(self, answer: str, query: str, knowledge: str) -> Dict[str, Any]:
        """
        验证回答质量
        
        Args:
            answer: 生成的回答
            query: 用户问题
            knowledge: 知识库内容
            
        Returns:
            质量评估结果
        """
        try:
            # 检查回答是否为空
            if not answer or len(answer.strip()) < 10:
                return {
                    "is_valid": False,
                    "score": 0,
                    "issues": ["回答太短或为空"],
                    "suggestion": "请提供更详细的回答"
                }
            
            # 检查是否包含无法回答的提示
            cannot_answer_keywords = ["无法回答", "不知道", "不了解", "没有相关信息", "抱歉"]
            if any(keyword in answer for keyword in cannot_answer_keywords):
                if knowledge:
                    # 有知识库但说无法回答，可能是质量问题
                    return {
                        "is_valid": True,
                        "score": 30,
                        "issues": ["回答表示无法回答，但知识库中有相关信息"],
                        "suggestion": "尝试重新组织回答，利用知识库内容"
                    }
                else:
                    # 确实没有相关知识
                    return {
                        "is_valid": True,
                        "score": 80,
                        "issues": [],
                        "suggestion": "回答诚实，但可以建议用户提供更多信息"
                    }
            
            # 检查回答长度是否合理
            answer_length = len(answer)
            if answer_length < 50:
                score = 60
                issues = ["回答可能过于简短"]
            elif answer_length > 500:
                score = 70
                issues = ["回答可能过于冗长"]
            else:
                score = 85
                issues = []
            
            # 检查是否包含关键词（如果知识库不为空）
            if knowledge:
                # 从知识库中提取关键词
                knowledge_keywords = self._extract_keywords(knowledge)
                query_keywords = self._extract_keywords(query)
                
                # 检查回答中是否包含相关关键词
                relevant_keywords = [kw for kw in knowledge_keywords + query_keywords if kw in answer]
                if relevant_keywords:
                    score += 10
                else:
                    issues.append("回答可能不够相关")
                    score -= 15
            
            # 最终评估
            return {
                "is_valid": score >= 50,
                "score": min(100, max(0, score)),
                "issues": issues,
                "suggestion": "回答质量良好" if score >= 70 else "建议优化回答"
            }
            
        except Exception as e:
            logger.error(f"回答质量验证失败: {e}")
            return {
                "is_valid": True,  # 默认通过验证
                "score": 60,
                "issues": ["质量验证过程出错"],
                "suggestion": "请人工检查回答质量"
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单的关键词提取：去除停用词，保留名词性词汇
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        
        # 中文分词简单实现（实际项目中可以使用jieba等分词库）
        words = []
        current_word = ""
        for char in text:
            if char.isalnum() or '\u4e00' <= char <= '\u9fff':
                current_word += char
            else:
                if current_word and current_word not in stop_words and len(current_word) > 1:
                    words.append(current_word)
                current_word = ""
        
        if current_word and current_word not in stop_words and len(current_word) > 1:
            words.append(current_word)
        
        return list(set(words))  # 去重


# 全局AI服务实例
_ai_service_instance = None

def get_ai_service(api_key: str = None, model: str = None, temperature: float = None) -> AIService:
    """
    获取AI服务实例（单例模式）
    
    Args:
        api_key: API密钥
        model: 模型名称
        temperature: 温度参数
        
    Returns:
        AI服务实例
    """
    global _ai_service_instance
    
    if _ai_service_instance is None:
        if not api_key:
            from config import DASHSCOPE_API_KEY
            api_key = DASHSCOPE_API_KEY
        
        if not model:
            from config import LLM_MODEL
            model = LLM_MODEL
        
        if temperature is None:
            from config import LLM_TEMPERATURE
            temperature = LLM_TEMPERATURE
        
        _ai_service_instance = AIService(api_key, model, temperature)
    
    return _ai_service_instance
