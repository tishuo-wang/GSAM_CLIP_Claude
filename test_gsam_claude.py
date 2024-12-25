import json
import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image

# Cloudinary 配置
CLOUD_NAME = "db1vryvyi"
API_KEY = "832756582328538"
API_SECRET = "s_Kgm1cPfrxLF3gkGAtPgQ3R_Zg"


def upload_image_to_cloudinary(image, cloud_name, api_key, api_secret):
    """
    使用 Cloudinary API 上传图片并返回图片 URL
    """
    # Cloudinary 上传 URL
    upload_url = f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload"

    # 将 PIL Image 转换为字节流
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_data = buffered.getvalue()

    # 请求数据
    files = {
        "file": ("image.png", image_data, "image/png")
    }
    data = {
        "upload_preset": "image_upload_url",  # 替换为实际的 Cloudinary 上传预设
    }

    # 发送 POST 请求到 Cloudinary
    response = requests.post(upload_url, data=data, files=files, auth=(api_key, api_secret))

    # 检查响应状态码
    if response.status_code == 200:
        # 返回图片的 URL
        return response.json()["secure_url"]
    else:
        # 打印错误信息并抛出异常
        raise Exception(f"Failed to upload image to Cloudinary: {response.text}")


def process_image_from_url(image_url):
    """
    从 Cloudinary 返回的图片 URL 下载图片并转换为 Base64 编码格式。
    """
    # 下载图片
    response = requests.get(image_url)
    if response.status_code == 200:
        image_data = response.content
    else:
        raise Exception(f"Failed to download image from URL: {response.status_code}, {response.text}")

    # 将图片数据编码为 Base64
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    media_type = "image/png"  # 图片格式为PNG
    return media_type, image_base64


def call_claude_api(user_input, image):
    """
    调用 Claude 的 API 接口，同时支持文本和图片输入。
    """
    try:
        # 如果有图片上传，先上传到 Cloudinary 并获取图片 URL
        image_url = upload_image_to_cloudinary(image, CLOUD_NAME, API_KEY, API_SECRET) if image else None
        print(f"Image URL: {image_url}")  # 打印图片 URL 以验证

        # 如果有图片 URL，处理图片数据
        if image_url:
            media_type, image_base64 = process_image_from_url(image_url)

        # 构建 API 请求消息
        headers = {
            "x-api-key": "sk-y4WrOWfbLMD6QzIccF7Z44Wcvw71XGjpRUsI10mSbvzxnl0p",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        messages = [{"role": "user", "content": user_input}]  # 添加用户文本消息

        if image_url:  # 添加图片消息
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        }
                    ],
                }
            )

        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "stream": False,
            "messages": messages,
        }

        # 打印请求数据
        print(f"Request Data: {json.dumps(data, indent=4)}")

        # 发送请求
        response = requests.post(
            "https://api.openai-proxy.org/anthropic/v1/messages",
            headers=headers,
            json=data,
        )

        # 打印 API 响应
        print(f"API Response: {response.json()}")

        # 检查返回数据
        if response.status_code == 200:
            response_data = response.json()
            if "content" in response_data:
                return response_data["content"][0]["text"]
            else:
                return f"API did not return a content field: {response_data}"
        else:
            return f"Failed to call API: {response.status_code}, {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"


from test_GroundedSAM import GroundedSAMWrapper  # 导入 GroundedSAM 类


if __name__ == "__main__":
    # 配置
    config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "groundingdino_swint_ogc.pth"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    image_path = "./assets/test/CLIP_test/00.jpg"
    text_prompt = "package. box. bottle. "
    output_dir = "outputs"
    box_threshold = 0.35
    text_threshold = 0.35
    device = "cuda"

    # 初始化封装类
    grounded_sam = GroundedSAMWrapper(config_file, grounded_checkpoint, sam_checkpoint, device)

    # 获取bounding boxes
    boxes = grounded_sam.get_bounding_boxes(image_path, text_prompt, box_threshold, text_threshold)
    print("Bounding boxes:", boxes)

    # 合并裁剪后的图像
    merged_image = grounded_sam.merge_cropped_boxes(image_path, boxes, output_path=None)

    text = "尽可能用中文以json格式输出图片中从左到右所有主要物体的信息,包括商品名称、分类、tag、口味等属性,只输出物品信息,自然语言的部分尽量使用中文,不要输出多余的文字,注意是从左到右"

    # 调用Claude API并获取返回结果
    response = call_claude_api(text, merged_image)
    

    # 打印Claude API返回的结果
    print("结果--------------------------------------------------\n",response)

    merged_image.show()
