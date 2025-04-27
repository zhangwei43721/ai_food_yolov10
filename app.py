import gradio as gr
import cv2
import tempfile
import os
import sys
import logging
from ultralytics.models.yolov10 import YOLOv10

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    try:
        logger.info(f"开始推理 model_id={model_id}, size={image_size}, conf={conf_threshold}")
        
        # 检查是否有本地模型文件
        if model_id == "yolov10n" and os.path.exists("yolov10n.pt"):
            logger.info("使用本地yolov10n.pt模型文件")
            model = YOLOv10("yolov10n.pt")
        else:
            # 否则尝试从默认位置加载
            model_name = f'yolov10{model_id[7:]}' if model_id.startswith('yolov10') else model_id
            logger.info(f"尝试加载模型: {model_name}")
            model = YOLOv10(model_name)
        
        detect_results_text = ""
        
        if image:
            logger.info(f"处理图像")
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
            # 获取检测结果文本
            for result in results:
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        cls_id = int(box.cls.item())
                        cls_name = result.names[cls_id]
                        conf = box.conf.item()
                        detect_results_text += f"检测到 {cls_name}，置信度: {conf:.2f}\n"
            
            annotated_image = results[0].plot()
            return annotated_image[:, :, ::-1], None, detect_results_text
        else:
            logger.info(f"处理视频")
            video_path = tempfile.mktemp(suffix=".webm")
            with open(video_path, "wb") as f:
                with open(video, "rb") as g:
                    f.write(g.read())

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_video_path = tempfile.mktemp(suffix=".webm")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))
            
            frame_count = 0
            detected_objects = {}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
                
                # 收集该帧的检测结果
                if frame_count % 10 == 0:  # 每10帧记录一次，避免重复太多
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls_id = int(box.cls.item())
                                cls_name = result.names[cls_id]
                                conf = box.conf.item()
                                if cls_name in detected_objects:
                                    if conf > detected_objects[cls_name]:
                                        detected_objects[cls_name] = conf
                                else:
                                    detected_objects[cls_name] = conf
                
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

            cap.release()
            out.release()
            
            # 生成视频检测结果文本
            for cls_name, conf in detected_objects.items():
                detect_results_text += f"检测到 {cls_name}，最高置信度: {conf:.2f}\n"

            return None, output_video_path, detect_results_text
    except Exception as e:
        logger.error(f"推理错误: {e}", exc_info=True)
        return None, None, f"错误: {str(e)}"


def yolov10_inference_for_examples(image, model_path, image_size, conf_threshold):
    try:
        logger.info(f"示例推理 model_path={model_path}")
        
        # 检查是否有本地模型文件
        if model_path == "yolov10n" and os.path.exists("yolov10n.pt"):
            logger.info("使用本地yolov10n.pt模型文件")
            model = YOLOv10("yolov10n.pt")
        else:
            # 否则尝试从默认位置加载
            model_name = f'yolov10{model_path[7:]}' if model_path.startswith('yolov10') else model_path
            logger.info(f"尝试加载模型: {model_name}")
            model = YOLOv10(model_name)
            
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1]
    except Exception as e:
        logger.error(f"示例推理错误: {e}", exc_info=True)
        return None


def app():
    with gr.Blocks() as blocks:
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n",
                        "yolov10s",
                        "yolov10m",
                        "yolov10b",
                        "yolov10l",
                        "yolov10x",
                    ],
                    value="yolov10n",  # 默认使用yolov10n
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="Detect Objects")
                status_info = gr.Textbox(label="状态信息", value="准备就绪")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)
                detect_results = gr.Textbox(label="检测结果", value="", lines=10)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            status = gr.update(value=f"切换到{input_type}模式")

            return image, video, output_image, output_video, status

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video, status_info],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            try:
                logger.info(f"运行推理: input_type={input_type}, model_id={model_id}")
                status = f"开始处理 {input_type}, 模型: {model_id}, 尺寸: {image_size}, 置信度: {conf_threshold}"
                if input_type == "Image":
                    img_result, vid_result, detect_text = yolov10_inference(image, None, model_id, image_size, conf_threshold)
                    status = "图像处理完成" if img_result is not None else "图像处理失败"
                    return img_result, vid_result, detect_text, status
                else:
                    img_result, vid_result, detect_text = yolov10_inference(None, video, model_id, image_size, conf_threshold)
                    status = "视频处理完成" if vid_result is not None else "视频处理失败"
                    return img_result, vid_result, detect_text, status
            except Exception as e:
                logger.error(f"运行推理错误: {e}", exc_info=True)
                return None, None, f"错误: {str(e)}", f"错误: {str(e)}"


        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video, detect_results, status_info],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
            ],
            fn=yolov10_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )
    return blocks

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app_interface = app()

if __name__ == '__main__':
    try:
        logger.info("启动Gradio应用...")
        gradio_app.launch(server_name="127.0.0.1", server_port=7862)
    except Exception as e:
        logger.error(f"启动Gradio应用失败: {e}", exc_info=True)
