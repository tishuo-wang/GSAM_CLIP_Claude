import argparse
import os
import copy
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GroundedSAMWrapper:
    def __init__(self, config_file, grounded_checkpoint, sam_checkpoint, device="cuda"):
        # 初始化配置
        self.config_file = config_file
        self.grounded_checkpoint = grounded_checkpoint
        self.sam_checkpoint = sam_checkpoint
        self.device = device

        # 加载模型
        self.grounded_model = self.load_model().to(self.device)  # 确保Grounding DINO模型在目标设备上
        self.sam_predictor = self.load_sam_predictor()

    def load_model(self):
        # 加载Grounding DINO模型
        args = SLConfig.fromfile(self.config_file)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        model.eval()
        return model

    def load_sam_predictor(self):
        # 加载SAM模型
        predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        return predictor

    def load_image(self, image_path):
        # 加载图片
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        image = image.to(self.device)  # 确保图片张量在目标设备上
        return image_pil, image

    def get_grounding_output(self, image, caption, box_threshold, text_threshold, with_logits=True):
        # 获取Grounding DINO模型的输出
        
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."

        image = image.to(self.device)  # 确保图片张量在目标设备上

        with torch.no_grad():
            outputs = self.grounded_model(image[None], captions=[caption])

        logits = outputs["pred_logits"].to(self.device).sigmoid()[0]  # 确保logits在目标设备上
        boxes = outputs["pred_boxes"].to(self.device)[0]  # 确保boxes在目标设备上

        # 筛选输出
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        # 获取预测的短语
        tokenlizer = self.grounded_model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt.to(self.device), pred_phrases

    def predict(self, image_path, text_prompt, box_threshold, text_threshold, output_dir):
        # 主预测逻辑
        image_pil, image = self.load_image(image_path)
        os.makedirs(output_dir, exist_ok=True)
        image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

        boxes_filt, pred_phrases = self.get_grounding_output(image, text_prompt, box_threshold, text_threshold)

        # 初始化SAM预测器
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_cv)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(self.device)  # 确保转换后的box在目标设备上
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(self.device)

        if transformed_boxes.numel() == 0:
            print("未检测到物体")
        else:
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )

            # 绘制输出图像
            plt.figure(figsize=(10, 10))
            plt.imshow(image_cv)
            for mask in masks:
                self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                self.show_box(box.numpy(), plt.gca(), label)

            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, "grounded_sam_output.jpg"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )

            self.save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)

    def save_mask_data(self, output_dir, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1]
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
            json.dump(json_data, f)
            
    def visualize_prediction(self, image_path, text_prompt, box_threshold, text_threshold):
        # 加载并处理图像
        image_pil, image = self.load_image(image_path)
        
        # 获取Grounding DINO的输出
        boxes_filt, pred_phrases = self.get_grounding_output(image, text_prompt, box_threshold, text_threshold)

        # 初始化SAM预测器
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_cv)

        # 获取图像尺寸
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(self.device)  # 调整为实际图像坐标
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(self.device)

        if transformed_boxes.numel() == 0:
            print("未检测到物体")
        else:
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )

            # 可视化分割结果
            plt.figure(figsize=(10, 10))
            plt.imshow(image_cv)

            # 绘制每个物体的分割mask
            for mask in masks:
                self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

            # 绘制每个物体的预测框和标签
            for box, label in zip(boxes_filt, pred_phrases):
                self.show_box(box.numpy(), plt.gca(), label)

            plt.axis('off')
            plt.show()  # 直接展示，而不是保存
    
    def get_bounding_boxes(self, image_path, text_prompt, box_threshold, text_threshold):
        # 获取符合描述的bounding box信息
        _, image = self.load_image(image_path)
        boxes_filt, _ = self.get_grounding_output(image, text_prompt, box_threshold, text_threshold)
        
        # 获取图像尺寸，转换坐标为实际图像坐标
        image_pil = Image.open(image_path)
        width, height = image_pil.size
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([width, height, width, height]).to(self.device)  # 调整为实际图像坐标
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        return boxes_filt.cpu().numpy().tolist()

    def merge_cropped_boxes(self, image_path, boxes, output_path=None):
        # 加载图像
        image_pil = Image.open(image_path)
        
        cropped_images = []
        total_width = 0
        max_height = 0
        
        # 按照x1坐标排序boxes（从左到右）
        sorted_boxes = sorted(boxes, key=lambda box: box[0])
        
        # 裁剪图像并记录总宽度和最大高度
        for box in sorted_boxes:
            x1, y1, x2, y2 = box  # 正确获取 bounding box 的坐标
            
            # 确保坐标顺序是正确的
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 裁剪图像
            cropped_image = image_pil.crop((x1, y1, x2, y2))
            cropped_images.append((cropped_image, x1))  # 记录图像和x1位置
        
        # 按照x1坐标对裁剪的图像进行排序
        cropped_images.sort(key=lambda item: item[1])  # 使用x1位置进行排序

        # 创建合并后的图像
        total_width = sum(cropped_image.width for cropped_image, _ in cropped_images)
        max_height = max(cropped_image.height for cropped_image, _ in cropped_images)
        merged_image = Image.new('RGB', (total_width, max_height))
        
        # 将每个裁剪图像按顺序粘贴到合并图像中
        current_x = 0
        for cropped_image, _ in cropped_images:
            merged_image.paste(cropped_image, (current_x, 0))
            current_x += cropped_image.width
        
        # 如果提供了输出路径，则保存图像
        if output_path:
            merged_image.save(output_path)
        
        return merged_image



if __name__ == "__main__":
    # 配置
    config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "groundingdino_swint_ogc.pth"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    image_path = "./assets/test/kouzhao.jpg"
    text_prompt = "package. box. bottle. "
    output_dir = "outputs"
    box_threshold = 0.35
    text_threshold = 0.35
    device = "cuda"

    # 初始化封装类
    grounded_sam = GroundedSAMWrapper(config_file, grounded_checkpoint, sam_checkpoint, device)

    # 执行预测
    grounded_sam.predict(image_path, text_prompt, box_threshold, text_threshold, output_dir)
    grounded_sam.visualize_prediction(image_path, text_prompt, box_threshold, text_threshold)
    
    
    # boxes = grounded_sam.get_bounding_boxes(image_path, text_prompt, box_threshold, text_threshold)
    # print("Bounding boxes:", boxes)
    # merged_image = grounded_sam.merge_cropped_boxes(image_path, boxes, output_path=None)
    # merged_image.show()

