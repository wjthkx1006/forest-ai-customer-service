"""
上下文处理器 - 智能处理对话上下文和指代关系
"""
from typing import List, Dict, Tuple, Optional
from .logger import logger


class ContextProcessor:
    """上下文处理器，负责智能分析对话上下文"""
    
    def __init__(self):
        self.reference_keywords = ["它", "这个", "那个", "上面说的", "之前提到的", "您说的", "您提到的"]
        self.question_keywords = ["怎么", "如何", "为什么", "什么", "哪些", "哪个", "多少", "多久"]
        self.pronouns = ["它", "他", "她", "这", "那", "这个", "那个", "这些", "那些"]
    
    def analyze_conversation_context(self, messages: List[Dict], current_query: str) -> Dict[str, any]:
        """
        分析对话上下文
        
        Args:
            messages: 对话历史
            current_query: 当前用户问题
            
        Returns:
            上下文分析结果
        """
        if len(messages) <= 2:
            return {
                "has_context": False,
                "referenced_topic": None,
                "topic_continuation": False,
                "clarification_needed": False,
                "summary": "这是第一次对话"
            }
        
        # 获取最近对话历史
        recent_messages = messages[-10:]  # 最多10条消息
        user_messages = [msg for msg in recent_messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in recent_messages if msg.get("role") == "assistant"]
        
        # 分析当前查询
        query_analysis = self._analyze_query(current_query)
        
        # 检查是否引用之前的话题
        referenced_topic = self._find_referenced_topic(current_query, recent_messages)
        
        # 检查是否是话题的延续
        topic_continuation = self._check_topic_continuation(current_query, recent_messages)
        
        # 检查是否需要澄清
        clarification_needed = self._check_clarification_needed(current_query, recent_messages)
        
        # 生成对话摘要
        conversation_summary = self._generate_conversation_summary(recent_messages)
        
        return {
            "has_context": len(recent_messages) > 2,
            "referenced_topic": referenced_topic,
            "topic_continuation": topic_continuation,
            "clarification_needed": clarification_needed,
            "query_analysis": query_analysis,
            "recent_user_questions": [msg.get("content", "") for msg in user_messages[-3:]],
            "recent_assistant_answers": [msg.get("content", "") for msg in assistant_messages[-3:]],
            "summary": conversation_summary
        }
    
    def _analyze_query(self, query: str) -> Dict[str, any]:
        """分析查询语句"""
        query_lower = query.lower()
        
        # 检查是否包含指代词
        has_reference = any(pronoun in query_lower for pronoun in self.pronouns)
        
        # 检查是否包含疑问词
        has_question_word = any(keyword in query_lower for keyword in self.question_keywords)
        
        # 提取可能的主题词
        topic_keywords = self._extract_topic_keywords(query)
        
        return {
            "has_reference": has_reference,
            "has_question_word": has_question_word,
            "topic_keywords": topic_keywords,
            "query_type": self._determine_query_type(query)
        }
    
    def _find_referenced_topic(self, current_query: str, messages: List[Dict]) -> Optional[str]:
        """查找被引用的主题"""
        query_lower = current_query.lower()
        
        # 检查是否包含指代词
        for pronoun in self.reference_keywords:
            if pronoun in query_lower:
                # 在最近的助理回答中寻找可能被引用的主题
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # 提取助理回答中的主要主题
                        topic = self._extract_main_topic(content)
                        if topic:
                            return topic
                break
        
        return None
    
    def _check_topic_continuation(self, current_query: str, messages: List[Dict]) -> bool:
        """检查是否是话题的延续"""
        if len(messages) < 4:
            return False
        
        # 获取最近的主题
        recent_topics = []
        for msg in messages[-4:]:
            content = msg.get("content", "")
            topic = self._extract_main_topic(content)
            if topic:
                recent_topics.append(topic)
        
        # 检查当前查询是否与最近主题相关
        current_topic = self._extract_main_topic(current_query)
        if not current_topic:
            return False
        
        # 如果当前主题与最近主题有重叠，认为是话题延续
        for topic in recent_topics:
            if self._topics_related(current_topic, topic):
                return True
        
        return False
    
    def _check_clarification_needed(self, current_query: str, messages: List[Dict]) -> bool:
        """检查是否需要澄清"""
        query_lower = current_query.lower()
        
        # 检查是否包含模糊的指代
        if any(pronoun in query_lower for pronoun in self.pronouns):
            # 检查最近的对话中是否有多个可能被指代的主题
            recent_topics = []
            for msg in messages[-4:]:
                content = msg.get("content", "")
                topic = self._extract_main_topic(content)
                if topic:
                    recent_topics.append(topic)
            
            # 如果有多个不同主题，可能需要澄清
            if len(set(recent_topics)) > 1:
                return True
        
        # 检查是否是非常简短或模糊的问题
        if len(current_query.strip()) < 10 and any(word in query_lower for word in ["这个", "那个", "它"]):
            return True
        
        return False
    
    def _generate_conversation_summary(self, messages: List[Dict]) -> str:
        """生成对话摘要"""
        if len(messages) <= 2:
            return "对话刚开始"
        
        # 提取关键信息
        key_points = []
        topics_covered = set()
        
        for msg in messages[-6:]:  # 最近6条消息
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            if role == "user":
                topic = self._extract_main_topic(content)
                if topic and topic not in topics_covered:
                    key_points.append(f"用户询问了关于{topic}的问题")
                    topics_covered.add(topic)
            elif role == "assistant":
                # 提取助理回答中的关键信息
                key_info = self._extract_key_information(content)
                if key_info:
                    key_points.append(key_info)
        
        if not key_points:
            return "对话进行中"
        
        return "；".join(key_points[-3:])  # 最多3个关键点
    
    def _extract_topic_keywords(self, text: str) -> List[str]:
        """从文本中提取主题关键词"""
        # 简单的关键词提取
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "个"}
        
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
        
        return list(set(words))
    
    def _extract_main_topic(self, text: str) -> Optional[str]:
        """提取主要主题"""
        if not text or len(text) < 10:
            return None
        
        # 简单的主题提取逻辑
        topic_keywords = [
            "手机", "iphone", "苹果", "华为", "小米", "三星",
            "电池", "屏幕", "摄像头", "内存", "处理器",
            "价格", "质量", "保修", "验机", "二手"
        ]
        
        text_lower = text.lower()
        for keyword in topic_keywords:
            if keyword in text_lower:
                return keyword
        
        # 如果没有匹配的关键词，返回前几个关键词
        keywords = self._extract_topic_keywords(text)
        if keywords:
            return keywords[0]
        
        return None
    
    def _extract_key_information(self, text: str) -> Optional[str]:
        """从助理回答中提取关键信息"""
        if not text or len(text) < 20:
            return None
        
        # 简单的信息提取：取第一句话或前50个字符
        sentences = text.split('。')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 10:
                return first_sentence[:50] + ("..." if len(first_sentence) > 50 else "")
        
        return text[:50] + ("..." if len(text) > 50 else "")
    
    def _determine_query_type(self, query: str) -> str:
        """确定查询类型"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["怎么", "如何", "怎样"]):
            return "how_to"
        elif any(word in query_lower for word in ["为什么", "为何"]):
            return "why"
        elif any(word in query_lower for word in ["什么", "哪些"]):
            return "what"
        elif any(word in query_lower for word in ["哪个", "哪款"]):
            return "which"
        elif any(word in query_lower for word in ["多少", "多少钱", "价格"]):
            return "price"
        elif any(word in query_lower for word in ["好吗", "怎么样", "质量"]):
            return "evaluation"
        else:
            return "general"
    
    def _topics_related(self, topic1: str, topic2: str) -> bool:
        """判断两个主题是否相关"""
        if not topic1 or not topic2:
            return False
        
        # 主题分类
        categories = {
            "device": ["手机", "iphone", "苹果", "华为", "小米", "三星", "智能机"],
            "component": ["电池", "屏幕", "摄像头", "内存", "处理器", "芯片"],
            "purchase": ["价格", "多少钱", "购买", "买", "选购"],
            "quality": ["质量", "好坏", "优劣", "验机", "检测"],
            "usage": ["使用", "操作", "设置", "功能"],
            "service": ["保修", "售后", "维修", "退货"]
        }
        
        # 检查是否属于同一类别
        for category, keywords in categories.items():
            topic1_in_category = any(keyword in topic1.lower() for keyword in keywords)
            topic2_in_category = any(keyword in topic2.lower() for keyword in keywords)
            
            if topic1_in_category and topic2_in_category:
                return True
        
        # 检查是否有相同的子字符串
        if topic1.lower() in topic2.lower() or topic2.lower() in topic1.lower():
            return True
        
        return False
    
    def build_context_aware_prompt(self, query: str, context_analysis: Dict, knowledge: str) -> str:
        """构建上下文感知的提示"""
        
        context_info = ""
        if context_analysis["has_context"]:
            context_info = f"""
【对话上下文分析】
- 对话摘要：{context_analysis['summary']}
- 当前查询类型：{context_analysis['query_analysis']['query_type']}
"""
            
            if context_analysis["referenced_topic"]:
                context_info += f"- 引用的主题：{context_analysis['referenced_topic']}\n"
            
            if context_analysis["topic_continuation"]:
                context_info += "- 这是之前话题的延续\n"
            
            if context_analysis["clarification_needed"]:
                context_info += "- 注意：当前查询可能需要澄清\n"
        
        return f"""
# 角色设定
你是专业的二手手机客服助手，正在进行智能对话。

{context_info if context_info else "【对话上下文】\n这是对话的开始，没有历史上下文。\n"}

# 相关知识
{knowledge if knowledge else "知识库中没有找到与当前对话直接相关的内容。"}

# 用户当前问题
{query}

# 对话处理要求
1. **上下文理解**：仔细分析对话上下文，理解用户的真实意图
2. **指代解析**：如果用户使用了"它"、"这个"、"那个"等指代词，请根据上下文确定所指内容
3. **话题连贯**：保持对话的连贯性，如果当前问题是之前话题的延续，请自然衔接
4. **意图识别**：识别用户问题的深层意图，不仅仅是表面文字

# 回答策略
1. **直接回答**：不要添加自我介绍或刻意的开场白，直接回答问题
2. **自然衔接**：如果当前问题是之前话题的延续，请自然衔接
3. **指代明确**：如果用户使用了指代词，请在回答中明确所指内容
4. **简洁专业**：回答要简洁明了，专业准确
5. **自然表达**：使用自然的口语化表达，不要显得刻意
6. **价值提供**：即使问题不完全相关，也要尝试提供有价值的信息

# 回答格式
- 使用中文回答
- 语气亲切自然
- 适当分段，提高可读性
- 重点信息可以适当强调

请开始回答：
"""


# 全局上下文处理器实例
_context_processor_instance = None

def get_context_processor() -> ContextProcessor:
    """获取上下文处理器实例（单例模式）"""
    global _context_processor_instance
    
    if _context_processor_instance is None:
        _context_processor_instance = ContextProcessor()
    
    return _context_processor_instance
