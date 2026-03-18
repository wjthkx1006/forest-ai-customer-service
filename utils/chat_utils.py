"""
聊天相关的工具函数
"""
from typing import List, Dict, Any
from .logger import logger

# ---------- 敏感词过滤 ----------
SENSITIVE_KEYWORDS = [
    "政治", "政府", "共产党", "国民党", "领导人", "主席", "总理", "总书记",
    "军事", "军队", "武器", "枪支", "弹药", "炸弹", "导弹", "战争",
    "毒品", "吸毒", "贩毒", "赌博", "色情", "淫秽", "卖淫", "嫖娼",
    "翻墙", "VPN", "恐怖", "暴力", "血腥", "自杀", "邪教"
]


def contains_sensitive_content(text: str) -> bool:
    """检测文本是否包含敏感内容"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SENSITIVE_KEYWORDS)


def process_sensitive_content(query: str) -> Dict[str, Any]:
    """处理敏感内容"""
    logger.info("检测到敏感内容")
    return {
        "response": "抱歉，我无法回答这个问题。我专注于二手手机相关的咨询服务。",
        "show_human": True,
        "is_sensitive": True,
        "from_cache": False,
        "response_time": 0
    }


def process_special_questions(query: str, messages: List[Dict] = None) -> Dict[str, Any]:
    """处理特殊问题"""
    query_clean = query.strip().lower().replace("？", "").replace("?", "").replace(" ", "")

    if query_clean in ["你是谁", "你叫什么"]:
        logger.info("处理特殊问题: 身份识别")
        return {
            "response": "您好！我是森林二手手机店的智能客服助手，很高兴为您服务！",
            "show_human": False,
            "is_sensitive": False,
            "from_cache": False,
            "response_time": 0
        }
    
    # 处理上下文相关问题
    if any(phrase in query.lower() for phrase in ["我刚问了什么", "我刚才问了什么", "我刚刚问了什么", "我之前问了什么"]):
        logger.info("处理特殊问题: 上下文查询")
        
        if messages and len(messages) > 1:
            # 查找用户最近的问题
            user_questions = []
            for msg in messages[-10:]:  # 查看最近10条消息
                if msg.get("role") == "user":
                    content = msg.get("content", "").strip()
                    if content and content != query:  # 排除当前问题
                        user_questions.append(content)
            
            if user_questions:
                # 返回最近的问题
                recent_questions = user_questions[-3:]  # 最近3个问题
                questions_text = "、".join(recent_questions)
                response = f"您最近问了：{questions_text}"
            else:
                response = "您还没有问过其他问题呢。"
        else:
            response = "这是我们的第一次对话，您还没有问过其他问题。"
        
        return {
            "response": response,
            "show_human": False,
            "is_sensitive": False,
            "from_cache": False,
            "response_time": 0
        }
    
    return None


def handle_unrelated_questions(query: str = "", context: str = "") -> str:
    """处理无关问题（优化版：先自由发挥，再自然引导）"""
    # 这里返回一个标记，实际处理在app.py中实现
    return "__UNRELATED_QUESTION__"


def generate_unrelated_response(query: str, context_analysis: dict = None) -> str:
    """生成无关问题的智能响应（先自由发挥，再引导）"""
    
    # 根据查询类型生成不同的自由发挥响应
    query_lower = query.lower()
    
    # 常见无关问题类型及响应
    if any(word in query_lower for word in ["天气", "气温", "下雨", "晴天"]):
        free_response = "关于天气的问题，我理解您可能想了解出行安排。不过作为二手手机客服，我更擅长解答手机相关问题。"
    elif any(word in query_lower for word in ["美食", "吃饭", "餐厅", "好吃"]):
        free_response = "美食话题总是让人愉快！虽然我对手机更专业，但理解您对生活品质的关注。"
    elif any(word in query_lower for word in ["电影", "电视剧", "娱乐", "明星"]):
        free_response = "娱乐话题确实有趣。我发现很多用户在选购手机时也会考虑影音体验呢。"
    elif any(word in query_lower for word in ["体育", "运动", "比赛", "健身"]):
        free_response = "运动健康很重要！其实手机也可以成为运动的好帮手，比如记录运动数据。"
    elif any(word in query_lower for word in ["新闻", "时事", "热点"]):
        free_response = "关注时事是个好习惯。科技新闻也是我经常关注的方向。"
    elif any(word in query_lower for word in ["你好", "嗨", "hello", "hi"]):
        free_response = "您好！很高兴为您服务！"
    elif any(word in query_lower for word in ["谢谢", "感谢", "辛苦了"]):
        free_response = "不客气！很高兴能为您提供帮助。"
    else:
        free_response = "我理解您的问题。"
    
    # 自然引导到二手手机话题
    guidance = "说到手机，我注意到很多用户关心二手手机的性价比和可靠性。如果您对二手手机有任何疑问，比如如何验机、电池健康怎么看、或者选购建议，我都很乐意为您详细解答。"
    
    # 如果有上下文分析，可以更精准地引导
    if context_analysis and context_analysis.get("has_context"):
        recent_topics = context_analysis.get("recent_user_questions", [])
        if recent_topics:
            # 尝试联系到最近的对话话题
            last_topic = _extract_main_topic_from_list(recent_topics)
            if last_topic:
                guidance = f"回到我们刚才聊到的{last_topic}，如果您对这方面有更多疑问，或者想了解其他手机相关问题，我都可以帮您解答。"
    
    return f"{free_response}\n\n{guidance}"


def _extract_main_topic(text: str) -> str:
    """提取主要主题（简化版）"""
    if not text:
        return ""
    
    topic_keywords = [
        "手机", "iphone", "苹果", "华为", "小米", "三星",
        "电池", "屏幕", "摄像头", "内存", "处理器",
        "价格", "质量", "保修", "验机", "二手"
    ]
    
    text_lower = text.lower()
    for keyword in topic_keywords:
        if keyword in text_lower:
            return keyword
    
    return "手机"


def _extract_main_topic_from_list(texts: list) -> str:
    """从文本列表中提取主要主题"""
    for text in texts:
        topic = _extract_main_topic(text)
        if topic:
            return topic
    return "手机"
