import logging

def setup_logger(log_filepath):
    # 创建一个名为 "train_logger" 的 logger
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.DEBUG)

    # 创建一个用于将日志写入文件的文件处理器
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)

    # 创建一个用于将日志输出到控制台的控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个格式化器，用于配置日志消息的格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 将格式化器添加到文件处理器和控制台处理器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
