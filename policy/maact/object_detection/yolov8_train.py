from ultralytics import YOLO
import os

model = YOLO(r'object_detection/object_detection_ckpt/yolov8n.pt')

# 2. 训练模型
# 定义你的数据集 YAML 文件的路径
# 同样，如果你的脚本在 /my_project/ 目录下，而 data.yaml 在 /my_project/your_dataset/ 中
data_yaml_path = os.path.join('.', '/home/benson/projects/lerobot/object_detection/dataset2', 'data.yaml')

print(f"Starting training with data: {data_yaml_path}")

results = model.train(
   data=data_yaml_path,
   epochs=100,
   imgsz=(320, 256),
   batch=16,
   name='my_custom_detector_python', # 训练结果保存的文件夹名称
   device=0,
   patience=50,
   workers=8,
   pretrained=False,
   amp=False
)

print("Training complete!")

# 可选：训练完成后进行验证
metrics = model.val() # 默认在验证集上评估
print(f"mAP50-95 on validation set: {metrics.box.map}")
print(f"mAP50 on validation set: {metrics.box.map50}")

# 可选：保存训练结果（results 变量已包含）
model.save('custom_train.pt') # 训练完成后模型会自动保存，通常不需要手动调用