from ultralytics import YOLOv10

model = YOLOv10('train/weights/best.pt')
img_path = 'test3.jpg'
results = model(img_path, conf=0.25, iou=0.7)
results[0].show()

import torch
import cv2
import numpy as np

# # 定义一个钩子来拦截中间层输出
# feature_maps = []
# def hook(module, input, output):
#     feature_maps.append(output)
    
# # 在某个层上注册钩子，选择模型中的某个卷积层
# layer_to_hook = model.model.model[10]  # 假设第10层是我们感兴趣的层
# layer_to_hook.register_forward_hook(hook)

# img_path = 'test3.jpg'
# img = cv2.imread(img_path)  # 读取图像为 BGR 格式
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
# img = cv2.resize(img, (640, 640))  # 根据模型要求调整图像大小
# img = np.transpose(img, (2, 0, 1))  # 调整维度为 (C, H, W)
# img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # 转换为 Tensor 并添加 batch 维度

# # 进行推理（会触发钩子并保存中间层特征）
# with torch.no_grad():
#     results = model(img_tensor, conf=0.25, iou=0.7)

# # results = model("test3.jpg", conf=0.25, iou=0.7)
# results[0].show()

# names = results[0].names
# boxes = results[0].boxes.xyxy.tolist()
# classes = results[0].boxes.cls.tolist()
# confidences = results[0].boxes.conf.tolist()

# num = 0
# for box, cls, conf in zip(boxes, classes, confidences):
#     x1, y1, x2, y2 = box
#     tag = names[int(cls)]
#     score = int(round(conf,2)*100)
#     num += 1
#     print('-----xyxy, num=', num, int(x1), int(y1), int(x2), int(y2), 'tag=', tag, 'score=', score)
    

# # 提取特征向量
# if feature_maps:
#     feature_map = feature_maps[0].squeeze()  # 可以进一步处理这个特征图
#     C, H, W = feature_map.shape  # 获取特征图的通道数、高度和宽度
#     print('------------------------------------------------------')
#     # 根据输入图像和特征图的比例缩放 boxes
#     img_height, img_width = 640, 640  # 输入图像大小
#     scale_x = W / img_width  # x 轴缩放比例
#     scale_y = H / img_height  # y 轴缩放比例
#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = box
#         # 将 box 的坐标缩放到特征图的分辨率
#         x1_feat = int(x1 * scale_x)
#         y1_feat = int(y1 * scale_y)
#         x2_feat = int(x2 * scale_x)
#         y2_feat = int(y2 * scale_y)
        
#         # 确保坐标不超出特征图边界
#         x1_feat = max(0, min(x1_feat, W - 1))
#         y1_feat = max(0, min(y1_feat, H - 1))
#         x2_feat = max(0, min(x2_feat, W - 1))
#         y2_feat = max(0, min(y2_feat, H - 1))

#         # 从特征图中裁剪出对应区域
#         region_features = feature_map[:, y1_feat:y2_feat, x1_feat:x2_feat]  # 形状为 (C, h', w')
        
#         # 对区域特征进行池化或进一步处理
#         pooled_features = torch.mean(region_features, dim=[1, 2])  # 对 (h', w') 进行平均池化, 形状为 (C,)
        
#         # 输出每个框的特征向量
#         tag = names[int(classes[i])]
#         score = int(round(confidences[i], 2) * 100)
#         print(f"Box {i+1}: tag={tag}, score={score}, feature_vector_shape={pooled_features.shape}")

