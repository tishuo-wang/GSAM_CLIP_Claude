import torch
import open_clip
import cv2
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
from test_GroundedSAM import GroundedSAMWrapper
import numpy as np

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 初始化 Grounded SAM 模型
config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounded_checkpoint = "groundingdino_swint_ogc.pth"
sam_checkpoint = "sam_vit_h_4b8939.pth"
grounded_sam = GroundedSAMWrapper(config_file, grounded_checkpoint, sam_checkpoint, device)

# 初始化 CLIP 模型
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='./open_clip_pytorch_model.bin', device=device)
model.load_state_dict(torch.load("fine_tuned_clip_model.pth", map_location=device))
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 定义商品类别列表
product_classes = [
    "Lemon Drink",
    "Bottle",
    "南孚电池",
    "三得利无糖乌龙茶",
    "阿萨姆奶茶",
    "六神花露水",
    "乐事黄瓜味薯片",
    "乐事原味薯片",
    "东方树叶茉莉花茶",
    "可口可乐330ml",
    "Coca Cola"
]

# 输入图片和 Grounded SAM 参数
image_path = "./assets/test/CLIP_test/14.jpeg"
text_prompt = "bottle."
box_threshold = 0.35
text_threshold = 0.35

# 获取 Grounded SAM 的 bounding boxes
bounding_boxes = grounded_sam.get_bounding_boxes(image_path, text_prompt, box_threshold, text_threshold)
print("Bounding boxes:", bounding_boxes)

# 加载并处理图片
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(image_pil)
font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 60)


# 使用 CLIP 对每个检测到的区域进行分类
for box in bounding_boxes:
    x1, y1, x2, y2 = [int(coord) for coord in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    try:
        # 裁剪检测到的物体区域
        cropped_img = image_pil.crop((x1, y1, x2, y2))
        image_input = preprocess(cropped_img).unsqueeze(0).to(device)

        # 文本编码
        text_inputs = tokenizer(product_classes).to(device)

        # 编码图像和文本
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # 计算相似度
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (10 *  image_features @ text_features.T).softmax(dim=-1).squeeze(0).cpu().numpy()
        best_match_idx = similarity.argmax()
        best_match_class = product_classes[best_match_idx]
        best_match_score = similarity[best_match_idx]

        print(similarity)
        print(best_match_class, best_match_score)
        
        # 绘制检测框和标签
        label_text = f"{best_match_class} ({best_match_score:.2f})"
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, y1 - 10), label_text, font=font, fill=(255, 0, 0))

    except Exception as e:
        print(f"Error processing box {box}: {e}")
        continue

# 将 PIL 图像转换为 OpenCV 图像
image_result = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# 显示最终结果
cv2.namedWindow('GSAM+CLIP Detection', cv2.WINDOW_NORMAL)
cv2.imshow("GSAM+CLIP Detection", image_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


