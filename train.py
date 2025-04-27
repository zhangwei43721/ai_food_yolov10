#coding:utf-8
from ultralytics import YOLOv10
# 模型配置文件
model_yaml_path = 'ultralytics/cfg/models/v10/yolov10n.yaml'
# 数据集配置文件
data_yaml_path = 'C:/myself/hw_ai/ai_datasets/菜品目标检测数据集2600+/images/data.yaml'
# 预训练模型
pre_model_name = './yolov10n.pt'


if __name__ == '__main__':
    # 加载预训练模型
    model = YOLOv10(model_yaml_path).load(pre_model_name)
    # 训练模型 device = 0
    model.train(data=data_yaml_path, 
                epochs=100, 
                batch=4, 
                imgsz=640,
                name = 'train_v10',
                device='cpu')