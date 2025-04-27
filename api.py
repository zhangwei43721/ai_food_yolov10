from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import uuid
import cv2
import numpy as np
import shutil
import logging
from ultralytics.models.yolov10 import YOLOv10

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv10 Object Detection API", description="API for object detection using YOLOv10")

# 创建临时目录用于存储上传的图像和处理后的图像
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "yolov10_uploads")
RESULT_DIR = os.path.join(tempfile.gettempdir(), "yolov10_results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
@app.on_event("startup")
async def startup_event():
    global model
    try:
        if os.path.exists("yolov10n.pt"):
            logger.info("使用本地yolov10n.pt模型文件")
            model = YOLOv10("yolov10n.pt")
        else:
            logger.info("从默认位置加载yolov10n模型")
            model = YOLOv10("yolov10n")
        logger.info("YOLOv10模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

# 图像处理函数
def process_image(image_path, image_size=640, conf_threshold=0.25):
    try:
        # 读取图像
        if isinstance(image_path, np.ndarray):
            img = image_path
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
        
        # 执行推理
        results = model.predict(source=img, imgsz=image_size, conf=conf_threshold)
        
        # 获取检测结果文本
        detect_results = []
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls.item())
                    cls_name = result.names[cls_id]
                    conf = box.conf.item()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detect_results.append({
                        "class": cls_name,
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        # 绘制标注后的图像
        annotated_image = results[0].plot()
        
        return annotated_image, detect_results
    except Exception as e:
        logger.error(f"图像处理错误: {e}", exc_info=True)
        raise Exception(f"图像处理错误: {str(e)}")

# API端点：上传图像并进行检测
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), image_size: int = 640, conf_threshold: float = 0.25):
    try:
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        result_path = os.path.join(RESULT_DIR, f"{file_id}_result{file_extension}")
        
        # 保存上传的文件
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 处理图像
        annotated_image, detect_results = process_image(upload_path, image_size, conf_threshold)
        
        # 保存处理后的图像
        cv2.imwrite(result_path, annotated_image)
        
        return JSONResponse(content={
            "status": "success",
            "message": "检测完成",
            "result_image_url": f"/get_result_image/{file_id}{file_extension}",
            "detections": detect_results
        })
    except Exception as e:
        logger.error(f"检测错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

# API端点：获取处理后的图像
@app.get("/get_result_image/{file_name}")
async def get_result_image(file_name: str):
    result_path = os.path.join(RESULT_DIR, f"{os.path.splitext(file_name)[0]}_result{os.path.splitext(file_name)[1]}")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="图像不存在")
    return FileResponse(result_path)

# API端点：清理临时文件
@app.get("/cleanup")
async def cleanup():
    try:
        for file in os.listdir(UPLOAD_DIR):
            os.remove(os.path.join(UPLOAD_DIR, file))
        for file in os.listdir(RESULT_DIR):
            os.remove(os.path.join(RESULT_DIR, file))
        return {"status": "success", "message": "临时文件已清理"}
    except Exception as e:
        logger.error(f"清理错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# API端点：获取支持的模型
@app.get("/models")
async def get_models():
    return {
        "available_models": [
            {"id": "yolov10n", "name": "YOLOv10-Nano", "description": "最小最快的模型"},
            {"id": "yolov10s", "name": "YOLOv10-Small", "description": "小型模型"},
            {"id": "yolov10m", "name": "YOLOv10-Medium", "description": "中型模型"},
            {"id": "yolov10l", "name": "YOLOv10-Large", "description": "大型模型"},
            {"id": "yolov10x", "name": "YOLOv10-XLarge", "description": "超大型模型"}
        ]
    }

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "YOLOv10 API服务正常运行"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 