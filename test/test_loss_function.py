import torch
import torch.nn.functional as F
# 模型输出
logits = torch.tensor([[0.1, 0.8, 0.1,0,0],
                       [0.1, 3.0, 0.2]])
# 真实类别
print(f'logits shape:{logits.shape}')
targets = torch.tensor([1,2,4])
print(f'targets shape:{targets.shape}')

# 计算交叉熵损失
loss = F.cross_entropy(logits, targets)
print(loss)  # 输出将近似于1.277