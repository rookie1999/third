import os
import sys
import logging

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


def setup_logger(save_dir, name="Train"):
    """
    配置日志系统：同时输出到控制台和文件
    """
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log.txt')

    # 创建 Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重复打印

    # 清除之前的 handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # 格式化
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_run_dirs(log_root_dir="./logs"):
    """
    生成带时间戳的运行目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = os.path.join(log_root_dir, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    return run_dir, ckpt_dir, run_name


def save_train_loss_plot(save_dir, losses, epoch):
    """
    绘制并保存 Loss 曲线
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Train Loss')
    plt.title(f"Training Loss (Epoch {epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()


def kl_divergence(mu, logvar):
    """
    计算 KL 散度 (用于 VAE)
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld