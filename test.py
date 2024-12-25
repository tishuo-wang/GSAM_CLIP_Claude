from test_GroundedSAM import GroundedSAMWrapper  # 导入 GroundedSAM 类
from test_gsam_claude import call_claude_api

# if __name__ == "__main__":
#     # 配置
#     config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#     grounded_checkpoint = "groundingdino_swint_ogc.pth"
#     sam_checkpoint = "sam_vit_h_4b8939.pth"
#     image_path = "./assets/test/CLIP_test/00.jpg"
#     text_prompt = "package. box. bottle. "
#     output_dir = "outputs"
#     box_threshold = 0.35
#     text_threshold = 0.35
#     device = "cuda"

#     # 初始化封装类
#     grounded_sam = GroundedSAMWrapper(config_file, grounded_checkpoint, sam_checkpoint, device)

#     # 获取bounding boxes
#     boxes = grounded_sam.get_bounding_boxes(image_path, text_prompt, box_threshold, text_threshold)
#     print("Bounding boxes:", boxes)

#     # 合并裁剪后的图像
#     merged_image = grounded_sam.merge_cropped_boxes(image_path, boxes, output_path=None)

#     text = "尽可能用中文以json格式输出图片中从左到右所有主要物体的信息,包括商品名称、分类、tag、口味等属性,只输出物品信息,自然语言的部分尽量使用中文,不要输出多余的文字,注意是从左到右"

#     # 调用Claude API并获取返回结果
#     response = call_claude_api(text, merged_image)
    

#     # 打印Claude API返回的结果
#     print("结果--------------------------------------------------\n",response)

#     merged_image.show()
    
    
import os  # 用于文件和目录操作
import json  # 用于解析和处理JSON数据
from PIL import Image, ImageDraw, ImageFont  # 用于图像操作、绘制bounding box和标注
   
    
if __name__ == "__main__":
    # 配置
    config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "groundingdino_swint_ogc.pth"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    image_path = "./assets/test/bottles_change.jpg"
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

    text = (
    "请以严格的JSON格式输出图片中从左到右所有主要物体的信息，"
    "JSON结构固定为：{\"商品列表\": [{\"名称\": \"\", \"包装\": \"\", \"品牌\": \"\", \"口味\": \"\", \"特点\": []}]}。"
    "请严格按照上述JSON格式输出，不要添加任何多余的文字或自然语言描述。"
    "确保所有物体按从左到右的顺序依次排列在\"商品列表\"中。"
    )

    
    # 调用Claude API并获取返回结果
    response = call_claude_api(text, merged_image)
    print("结果--------------------------------------------------\n", response)
    
    # merged_image.show()

    # 加载原始图像
    original_image = Image.open(image_path)
    draw = ImageDraw.Draw(original_image)
    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 20)

    # 解析response中的JSON
    response_data = json.loads(response)
    item_list = response_data.get("商品列表", [])

    # 确保boxes和item_list长度一致
    boxes = sorted(boxes, key=lambda box: box[0])
    if len(boxes) != len(item_list):
        print("Warning: Bounding boxes and response data do not match in length!")

    # 绘制bounding box和标签
    for box, item in zip(boxes, item_list):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)  # 绘制边界框

        # 标签内容：品牌、名称和口味
        label = f"{item.get('品牌', '未知')} - {item.get('名称', '未知')} - {item.get('口味', '未知')}"
        draw.text((x1, y1 - 20), label, fill="blue", font=font)  # 在box顶部显示标签

    # 显示结果图像
    original_image.show()



