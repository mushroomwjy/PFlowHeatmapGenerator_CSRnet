import logging
import colorlog

# 设置控制台logger
logger = logging.getLogger()
console_handler = logging.StreamHandler()

# 设置控制台logger优先级最高为info
logger.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

# 颜色参数
log_colors_config = {
    'DEBUG': 'bold_white',
    'INFO': 'bold_green',
    'WARNING': 'bold_orange',
    'ERROR': 'bold_red',
    'CRITICAL': 'bold_red',

}

# 配置format，message设置为青色输出
formatter = colorlog.ColoredFormatter(
    fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : '
        '%(cyan)s%(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S',
    log_colors=log_colors_config
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
