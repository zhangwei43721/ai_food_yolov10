import requests
import base64
import json
import os
from PIL import Image
import io

# API服务地址
API_URL = "http://localhost:8000"

def test_detect_image():
    """测试图像检测API"""
    # 测试图像路径 - 替换为实际图像路径
    test_image_path = "test_image.jpg"  # 请确保此文件存在
    
    if not os.path.exists(test_image_path):
        print(f"测试图像 {test_image_path} 不存在，请提供一个有效的图像文件")
        return
    
    # 准备上传文件
    files = {
        'file': (os.path.basename(test_image_path), open(test_image_path, 'rb'), 'image/jpeg')
    }
    
    # 设置参数
    params = {
        'image_size': 640,
        'conf_threshold': 0.25
    }
    
    try:
        # 发送POST请求到检测API
        print("发送检测请求...")
        response = requests.post(f"{API_URL}/detect", files=files, params=params)
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            print("检测成功!")
            print(f"状态: {result['status']}")
            print(f"消息: {result['message']}")
            print(f"结果图片URL: {result['result_image_url']}")
            print("\n检测到的对象:")
            
            for i, detection in enumerate(result['detections'], 1):
                print(f"对象 {i}:")
                print(f"  类别: {detection['class']}")
                print(f"  置信度: {detection['confidence']:.2f}")
                print(f"  边界框: {detection['bbox']}")
            
            # 获取结果图像
            image_url = f"{API_URL}{result['result_image_url']}"
            img_response = requests.get(image_url)
            
            if img_response.status_code == 200:
                # 保存结果图像
                result_image_path = "detection_result.jpg"
                with open(result_image_path, 'wb') as f:
                    f.write(img_response.content)
                print(f"\n结果图像已保存到: {result_image_path}")
                
                # 显示图像（如果运行环境支持）
                try:
                    image = Image.open(io.BytesIO(img_response.content))
                    image.show()
                except Exception as e:
                    print(f"无法显示图像: {e}")
            else:
                print(f"获取结果图像失败: {img_response.status_code}")
        else:
            print(f"检测请求失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 关闭文件
        files['file'][1].close()

def test_health():
    """测试API健康状态"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print(f"健康检查: {result['status']} - {result['message']}")
        else:
            print(f"健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"健康检查错误: {e}")

def test_models():
    """测试获取支持的模型列表"""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            result = response.json()
            print("支持的模型:")
            for model in result['available_models']:
                print(f"  ID: {model['id']}")
                print(f"  名称: {model['name']}")
                print(f"  描述: {model['description']}")
                print()
        else:
            print(f"获取模型列表失败: {response.status_code}")
    except Exception as e:
        print(f"获取模型列表错误: {e}")

if __name__ == "__main__":
    # 先测试API是否健康
    test_health()
    print("\n")
    
    # 测试获取模型列表
    test_models()
    print("\n")
    
    # 测试图像检测
    test_detect_image() 