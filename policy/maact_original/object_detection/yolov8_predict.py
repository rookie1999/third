import cv2
from ultralytics import YOLO
import os

# 定义你训练好的模型路径
model_path = 'runs/detect/my_custom_detector_python8/weights/best.pt'

# 加载你的模型
model = YOLO(model_path)

# 定义你想要进行预测的图片路径
input_image_path = '/home/benson/.cache/huggingface/lerobot/benson-zhan/' \
                   'so100_act_large/images/observation.images.depth/episode_000088/frame_000177.png' # <<< 请务必修改为你的实际图片路径
input_image_path = "/home/benson/projects/lerobot/tmp/yolo_input_batch_3.png"
# 执行预测
# 'show=True' 参数会打开一个显示窗口来展示图片
# 'save=True' 参数会把带有预测结果的图片保存到磁盘上
target_width = 320
target_height = 256
results = model(input_image_path,
                imgsz=(target_height, target_width),  # 预测时使用的图片尺寸
                conf=0.25,  # 置信度阈值
                iou=0.7,    # NMS IoU 阈值
                device=0,   # 指定 GPU (或 'cpu')
                show=True,  # 显示带有预测结果的图片
                save=True   # 将预测后的图片保存到磁盘
               )

# 你也可以通过编程方式访问预测的详细信息：
# print(f"图片 {input_image_path} 的预测结果：")
# for r in results: # 'results' 是一个包含 Results 对象的列表，每个对象对应一张处理过的图片
#     # r.orig_img 是原始图片（NumPy 数组格式，如果需要）
#     # r.path 是原始图片的路径
#     # r.boxes 包含边界框的检测结果
#
#     boxes = r.boxes.xyxy.tolist()  # 边界框坐标 (x1, y1, x2, y2)
#     scores = r.boxes.conf.tolist() # 置信度分数
#     classes = r.boxes.cls.tolist() # 类别 ID
#     class_names = r.names        # 类别 ID 到名称的字典映射
#
#     print(f"  检测到 {len(boxes)} 个目标：")
#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = box
#         score = scores[i]
#         class_id = int(classes[i])
#         name = class_names[class_id]
#         print(f"    目标：{name}, 置信度：{score:.2f}, 边界框：[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
#
# print("\n预测完成。图片应该已经显示并保存了。")


# 对图片和检测框进行缩放后再可视化
print(f"图片 {input_image_path} 的预测结果：")
for r in results:
    # r.orig_img 是原始图片 (NumPy 数组格式)
    original_image = r.orig_img

    # 获取原始图片尺寸
    orig_h, orig_w = original_image.shape[:2]

    # 手动缩放图片
    scaled_image = cv2.resize(original_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    boxes = r.boxes.xyxy.tolist()  # 边界框坐标 (x1, y1, x2, y2)，此时仍是原始图片尺寸的坐标
    scores = r.boxes.conf.tolist()  # 置信度分数
    classes = r.boxes.cls.tolist()  # 类别 ID
    class_names = r.names  # 类别 ID 到名称的字典映射

    print(f"  检测到 {len(boxes)} 个目标：")
    for i, box in enumerate(boxes):
        x1_orig, y1_orig, x2_orig, y2_orig = box
        score = scores[i]
        class_id = int(classes[i])
        name = class_names[class_id]

        # 计算缩放因子
        scale_x = target_width / orig_w
        scale_y = target_height / orig_h

        # 缩放边界框坐标
        x1_scaled = int(x1_orig * scale_x)
        y1_scaled = int(y1_orig * scale_y)
        x2_scaled = int(x2_orig * scale_x)
        y2_scaled = int(y2_orig * scale_y)

        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        thickness = 2
        cv2.rectangle(scaled_image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, thickness)

        # 绘制标签和置信度
        label = f"{name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        # 标签背景框
        cv2.rectangle(scaled_image, (x1_scaled, y1_scaled - text_size[1] - 5),
                      (x1_scaled + text_size[0], y1_scaled - 5), color, -1)

        # 标签文字
        cv2.putText(scaled_image, label, (x1_scaled, y1_scaled - 5), font, font_scale, (0, 0, 0), font_thickness,
                    cv2.LINE_AA)

        print(
            f"    目标：{name}, 置信度：{score:.2f}, 原始边界框：[{x1_orig:.0f}, {y1_orig:.0f}, {x2_orig:.0f}, {y2_orig:.0f}], 缩放后边界框：[{x1_scaled:.0f}, {y1_scaled:.0f}, {x2_scaled:.0f}, {y2_scaled:.0f}]")

    # 显示缩放并绘制了边界框的图片
    cv2.imshow("Scaled Image with Detections", scaled_image)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭窗口

    # 保存缩放并绘制了边界框的图片
    output_dir = "yolo_scaled_detections"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_image_path)
    output_path = os.path.join(output_dir, f"scaled_detected_{base_name}")
    cv2.imwrite(output_path, scaled_image)
    print(f"缩放并绘制了边界框的图片已保存到: {output_path}")

print("\n预测和可视化完成。")