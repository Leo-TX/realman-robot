'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-30 17:29:24
Version: v1
File: 
Brief: 
'''
'''
系统环境: Windows10
Python版本: 3.7
PyTorch版本: 1.1.0
cuda: no
'''
import torch
import torch.nn.functional as F   # 激励函数的库    
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

# 定义全局变量
n_epochs = 10     # epoch 的数目
batch_size = 20  # 决定每次读取多少图片

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

# 定义训练集个测试集，如果找不到数据，就下载
train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = './data', train = False, download = True, transform = transform)

# 创建加载器
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0, shuffle = True)


# 建立一个四层感知机网络

class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__()    # 
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(784,512)  # 第一个隐含层(全连接层fc，或者叫线性层linear)
        self.relu1 = torch.nn.ReLU() # 使用 relu 激活函数
        self.fc2 = torch.nn.Linear(512,128)  # 第二个隐含层
        self.relu2 = torch.nn.ReLU() # 使用 relu 激活函数
        self.fc3 = torch.nn.Linear(128,10)   # 输出层
        self.softmax = torch.nn.Softmax(dim=1)   # 输出层使用 softmax 激活函数
    def forward(self,x):
        # 前向传播， 输入值：x, 返回值 x
        print(f'forward function is called')
        x = x.view(-1,28*28)       # 将一个多行的Tensor,拼接成一行，或者： x=torch.flatten(x,start_dim=1) 
        print(f'x shape:{x.shape}')
        x = self.fc1(x)  
        print(f'x shape:{x.shape}')
        x = self.relu1(x)   # 使用 relu 激活函数
        print(f'x shape:{x.shape}')
        x = self.fc2(x)    
        print(f'x shape:{x.shape}')
        x = self.relu2(x)   # 使用 relu 激活函数
        print(f'x shape:{x.shape}')
        x = self.fc3(x)   
        print(f'x shape:{x.shape}')
        x = self.softmax(x)  # 使用 softmax 激活函数
        print(f'x shape:{x.shape}')
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return x

# 声明感知器网络
model = MLP()
params = list(model.parameters())  # 获取模型的参数列表
for i, param in enumerate(params):
    print(f"Parameter {i}: {param.size()}")
model.train()
# 训练神经网络
def train():
    #定义损失函数和优化器
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        train_correct = 0.0
        for data,labels in train_loader:
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            output = model(data)    # 得到预测值
            print(f'output shape:{output.shape}')
            print(f'labels shape:{labels.shape}')
            loss = lossfunc(output,labels)  # 计算两者的误差
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item()*data.size(0)

            _,id=torch.max(output.data,1)
            train_correct+=torch.sum(id==labels.data)
            break
        train_loss = train_loss / len(train_loader.dataset)
        # 遍训练遍测试：训练集
        train_correct = train_correct / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}   Correct:{}%'.format(epoch + 1, train_loss, (100 * train_correct)))
        # 遍训练遍测试：测试集
        # test()
        break

model.eval() # 有的测试代码前面要加上 model.eval()，表示这是训练状态。如果没有 Batch Normalization 和 Dropout 方法，加和不加的效果是一样的。
# 在数据集上测试神经网络
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播,不需要计算梯度，能够节省一些内存空间
        for data, labels in test_loader:
            print(f'images shape:{data.shape}')
            print(f'labels shape:{labels.shape}')
            print(f'labels:{labels}')
            outputs = model(data)
            print(f'outputs shape:{outputs.shape}')
            _, predicted = torch.max(outputs.data, 1)
            print(f'predicted shape:{predicted.shape}')
            print(f'predicted:{predicted}')
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            break
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    return 100.0 * correct / total


# def test_1():
#     for epoch in range(n_epochs):
#         for data,target in train_loader:
#             print(f'data type:{type(data)}')
#             print(f'data shape:{data.shape}')
#             # print(f'data:{data}')
#             print(f'target:{target}')
#             # print(data[0])
#             print(f'data[0] type:{type(data[0])}')
#             print(f'data[0] shape:{data[0].shape}')
if __name__ == '__main__':
    train()

