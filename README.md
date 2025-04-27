# YOLOv10 对象检测 API 服务

这是一个基于YOLOv10和FastAPI的对象检测API服务，能够接收图像并返回检测结果。

## 功能特点

- 提供RESTful API接口进行图像对象检测
- 支持多种YOLOv10模型（nano/small/medium/large/xlarge）
- 返回带有标注的图像和JSON格式的检测结果
- 支持调整图像大小和置信度阈值

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 启动服务

```bash
# 启动API服务
python api.py
```

服务将在 http://localhost:8000 启动，并提供API文档界面（http://localhost:8000/docs）。

## API接口

### 1. 检测图像中的对象

**端点:** `/detect`

**方法:** POST

**参数:**
- `file`: 图像文件（必需）
- `image_size`: 图像大小（可选，默认640）
- `conf_threshold`: 置信度阈值（可选，默认0.25）

**响应:**
```json
{
  "status": "success",
  "message": "检测完成",
  "result_image_url": "/get_result_image/uuid.jpg",
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [10, 20, 100, 200]
    },
    ...
  ]
}
```

### 2. 获取检测结果图像

**端点:** `/get_result_image/{file_name}`

**方法:** GET

**响应:** 图像文件

### 3. 查看支持的模型

**端点:** `/models`

**方法:** GET

**响应:**
```json
{
  "available_models": [
    {
      "id": "yolov10n",
      "name": "YOLOv10-Nano",
      "description": "最小最快的模型"
    },
    ...
  ]
}
```

### 4. 健康检查

**端点:** `/health`

**方法:** GET

**响应:**
```json
{
  "status": "healthy",
  "message": "YOLOv10 API服务正常运行"
}
```

## 测试API

使用提供的测试脚本进行API测试：

```bash
# 先确保API服务已启动
python test_api.py
```

## 示例用法

### Python客户端示例

```python
import requests

# 上传图像进行检测
files = {'file': open('test_image.jpg', 'rb')}
params = {'image_size': 640, 'conf_threshold': 0.25}

response = requests.post('http://localhost:8000/detect', files=files, params=params)
result = response.json()

# 获取检测结果图像
image_url = f"http://localhost:8000{result['result_image_url']}"
img_response = requests.get(image_url)

# 保存结果图像
with open('result.jpg', 'wb') as f:
    f.write(img_response.content)

# 显示检测结果
for detection in result['detections']:
    print(f"类别: {detection['class']}, 置信度: {detection['confidence']:.2f}")
```

## 注意事项

- 服务默认使用YOLOv10n模型，您可以根据需要修改代码以使用其他模型
- 对于生产环境，建议配置适当的CORS策略和安全措施
- 临时文件会存储在系统的临时目录中，可以通过`/cleanup`端点清理
