import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# 加载 Faster R-CNN 模型 (ResNet-50 backbone, FPN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

# 加载图像
image = Image.open('/media/datadisk10tb/leo/projects/realman-robot/open_door/data/test/001.png').convert("RGB")

# 图像预处理
def transform(image):
    image = F.to_tensor(image)
    # 图像归一化 (根据 ImageNet 数据集的均值和标准差)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

image_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

# 进行推理
with torch.no_grad():
    detections = model(image_tensor)

# 打印模型变量 (可选)
# model_vars = vars(model)
# for name, value in model_vars.items():
#     print(f'name: {name}, type: {type(value)}')

# 处理检测结果
print(f'type of detections: {type(detections)}')  # 输出 detections 的类型
for detection in detections:
    print(f'detection:\n{detection}')
    boxes = detection['boxes']
    scores = detection['scores']
    labels = detection['labels']

    # 打印检测结果信息 (可选)
    print(f'type of boxes: {type(boxes)}')
    print(f'boxes:\n{boxes}')
    print(f'type of scores: {type(scores)}')
    print(f'scores:\n{scores}')
    print(f'type of labels: {type(labels)}')
    print(f'labels:\n{labels}')

    #  根据置信度阈值过滤检测结果
    high_confidence_indices = scores > 0.5  # 例如，置信度阈值设置为 0.5
    filtered_boxes = boxes[high_confidence_indices]
    # ... 处理过滤后的边界框 ...