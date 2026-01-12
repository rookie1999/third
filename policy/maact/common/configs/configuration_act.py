import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from ultralytics import YOLO


def create_yolo_detector(ckpt_path: str, device: torch.device) -> YOLO:
    """
    创建一个并配置 YOLO 检测器实例。
    """
    logging.info(f"YOLO Factory: Loading YOLO model from {ckpt_path}")
    yolo_model = YOLO(ckpt_path)
    yolo_model.eval()  # 确保它处于评估模式
    yolo_model.to(device)  # 将模型移动到指定设备
    logging.info("YOLO Factory: YOLO model loaded and set to eval mode.")
    return yolo_model

# 假设您有一个地方存储了已经创建好的 YOLO 模型实例，以便在多个地方复用
_global_yolo_detector_instance = None

def get_yolo_detector(ckpt_path: str, device: torch.device):
    """
    获取一个全局的 YOLO 检测器实例。如果尚未创建，则创建它。
    """
    global _global_yolo_detector_instance
    if _global_yolo_detector_instance is None:
        _global_yolo_detector_instance = create_yolo_detector(ckpt_path, device)
    return _global_yolo_detector_instance

def perform_yolo_detection(image_tensor: torch.Tensor, ckpt_path: str, device: torch.device,
                           imgsz: tuple | int = 320,  # 预测时使用的图片尺寸，可以是一个整数（如 640）或 (高, 宽) 元组
                           conf_threshold: float = 0.2,  # 置信度阈值
                           ):
    yolo_model = get_yolo_detector(ckpt_path, device)
    if yolo_model is None:
        logging.warning("YOLO detection skipped: Model not loaded or checkpoint path is empty.")
        return None

    image_tensor = image_tensor.to(device)

    results = yolo_model(
        image_tensor,
        imgsz=imgsz,
        conf=conf_threshold,
        verbose=False,
    )

    if False:
        output_dir = "tmp"
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"YOLO 检测结果可视化将保存到: {output_dir}")

        for i, result in enumerate(results):
            # 获取原始图像（用于可视化）
            # results.orig_img 返回的是 NumPy 数组 (H, W, C), BGR 格式
            orig_img_np = result.orig_img

            # 将 BGR 转换为 RGB 以便 matplotlib 正确显示（可选，如果用 cv2.imwrite 则不需要）
            img_rgb = cv2.cvtColor(orig_img_np, cv2.COLOR_BGR2RGB)

            # 保存原始图像
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            original_img_path = os.path.join(output_dir,
                                             f"yolo_frame_1_{timestamp}.png")
            # cv2.imwrite(original_img_path, img_rgb)
            # logging.info(f"YOLO 输入原始图像已保存到: {original_img_path}")

            # 绘制边界框并保存结果图像
            # result.plot() 会在图像上绘制检测到的边界框
            # 它会返回一个 NumPy 数组，通常是 RGB 格式
            img_with_boxes = result.plot()

            # 如果 result.plot() 返回的是 RGB，而 cv2.imwrite 需要 BGR，则需要转换
            # img_with_boxes_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

            detected_img_path = os.path.join(output_dir,
                                             f"yolo_frame_2_{timestamp}.png")
            img_with_boxes_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
            cv2.imwrite(detected_img_path, img_with_boxes_bgr)  # 大多数情况下直接保存即可
            logging.info(f"YOLO 检测结果图像已保存到: {detected_img_path}")

            # 记录检测到的物体数量
            num_boxes = len(result.boxes)
            logging.info(f"批次 {i} 的 YOLO 检测结果：检测到 {num_boxes} 个物体。")
            if num_boxes == 0:
                logging.warning(f"注意：批次 {i} 未检测到任何物体。请检查图像内容或调整置信度/IoU阈值。")
    # --- 可视化部分结束 ---

    return results


@dataclass
class SpeedACTConfig:
    # --- 核心超参 ---
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100
    n_obs_steps: int = 1
    dropout: float = 0.1

    # --- [关键修复] 添加缺失的 Transformer 属性 ---
    feedforward_activation: str = "relu"  # <--- 必须添加这个
    pre_norm: bool = False  # <--- 建议同时添加这个，base_act 经常用到

    # --- 视觉相关 ---
    # 注意：image_features 在 dataclass 初始化时通常为 None，测试时手动注入或由 factory 处理
    image_features: Optional[Dict[str, Tuple[int, int, int]]] = None
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: Optional[str] = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # --- 机器人状态 ---
    # 建议给个默认值 None，避免实例化时报错
    robot_state_feature: Optional[object] = None
    action_feature: Optional[object] = None
    env_state_feature: Optional[object] = None

    # --- VAE 相关 ---
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0

    # --- SpeedACT 特有配置 ---
    main_camera: str = "camera_front"  # 给个默认值方便测试
    use_optical_flow: bool = True
    object_detection_ckpt_path: Optional[str] = None

    optical_flow_map_height: int = 256
    optical_flow_map_width: int = 320
    cropped_flow_h: int = 64
    cropped_flow_w: int = 64

    # 融合模块参数
    num_speed_categories: int = 3
    speed_loss_weight: float = 0.1
    speed_category_key: str = "speed_category"
    speed_logits_key: str = "speed_logits_internal"
    is_flow_valid_batch_key: str = "is_flow_valid_batch_internal"

    # 预融合 Dropout
    pre_fusion_dropout: float = 0.01