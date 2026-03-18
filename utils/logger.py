import logging
import os
from datetime import datetime

def setup_logger(name: str = "ai_customer_service"):
    """设置日志记录器"""
    # 创建logs目录
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # 生成日志文件名
    log_file = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # 配置日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建全局日志记录器
logger = setup_logger()

