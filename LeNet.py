'''
@author: Bi Xiong
@time: 2024/4/1 22:16
'''
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import  matplotlib as plt

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

'(1)setting'
random_seed = 1
lr = 0.001
epochs = 100
batch_size = 100
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'(2)dataset'
'使用 transforms.Compose([]) 函数将不同的数据预处理操作按顺序打包为一个整体，然后作用于输入图像'
transforms = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='data',train=True,download=True,transform=transforms)
test_dataset = datasets.MNIST(root='data',train=False,download=True, transform=transforms)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)

'(3)检验训练集数据集'
for images,labels in train_loader:
    print('image batch dimension:',images.shape,
          'label batch dimension:',labels.shape)
    break

'（4）LeNet model'
class LeNet(nn.Module):
    '因为MNIST数据集中的图像为灰度图像，所以令判定函数 grayscale = True，此时这些图像的输入通道数为 1'
    def __init__(self,num_classes,grayscale = True):
        super(LeNet, self).__init__()
        if grayscale == True:
            in_channels = 1
        else:
            in_channels = 3

        self.feature_stage = nn.Sequential(
            #  输入形状（32，32，1）， 输出形状（28,28,6）
            nn.Conv2d(in_channels=in_channels,out_channels=6,kernel_size=5,stride=1),

            #  输入形状为（28，28，6），输出形状（14,14,6）
            nn.MaxPool2d(kernel_size=2,stride=2),

            #  输入形状为（14,14,6），输出形状为（10,10,16）
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),

            #  输入形状为（10,10,16），输出形状为（5,5,16）
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.classification_stage = nn.Sequential(

            nn.Linear(in_features=16*5*5,out_features=120),

            nn.Linear(in_features=120,out_features=84),

            nn.Linear(in_features=84,out_features=num_classes)
        )

    def forward(self,x):

        x = self.feature_stage(x)

        '将卷积层的输出（实际为4维tensor（batch_size,H,W,C），但视为3维tensor（H,W,C）） 转换为  全连接层的输入（实际为2维tensor（batch_size，in_features），但视为1维向量）'
        '其中in_features表示 全连接层的输入特征数 / 每个特征图（对应3维tensor）被展平为1维向量所具有的长度。 '
        '并且，in_features = H×W×C'
        x = x.view(100,16*5*5)

        '当2维tensor x 输入最后一个全连接层后，得到模型的原始输出/原始输出分数 logits'
        logits = self.classification_stage(x)

        '使用 激活函数softmax() 在 logits 的第二个维度上 将 logits 转换为 每个类别的预测概率，即probability'
        probability = F.softmax(logits, dim=1)

        return logits,probability

'调用 设置随机种子的函数， 用来指定随机种子的值'
torch.manual_seed(random_seed)
model = LeNet(num_classes,grayscale=True)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

'（5）定义准确率函数'
def accuracy(model,data_loader,device):
    correct_pred, num_examples = 0, 0
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        '将图像数据images输入到模型中，得到模型的原始输出/原始输出分数 logits 、每个类别的预测概率值probability'
        logits, probability= model(images)
        _, predicted_labels = torch.max(probability, dim=1)
        correct_pred += (predicted_labels == labels).sum()
        num_examples += labels.size(0)
    return correct_pred / num_examples *100

'（6）训练LeNet'
start_time = time.time()
for epoch in range(epochs):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        '前向传播'
        logits, probability = model(images)
        '计算损失'
        loss = F.cross_entropy(logits, labels)
        '梯度清零'
        optimizer.zero_grad()
        '反向传播'
        loss.backward()
        '更新权重参数'
        optimizer.step()
        '打印 当前批次索引/batch_idx 所对应的训练日志'
        if batch_idx % 50 == 0:
            print('Epoch: %03d | %03d   Batch: %03d | %03d   Loss：%.4f' % (epoch+1, epochs, batch_idx, len(train_loader), loss))
            print('Train accuracy: %.2f%%' % accuracy(model, train_loader, device))

    '计算训练集上 完成一次训练（即在一个epoch数下，遍历完所有的 批次索引/batch_idx 数）所花的时间'
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
'计算训练集上 完成所有次训练（即遍历完所有的epoch数） 所花的时间'
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))


'（7）测试LeNet'
with torch.no_grad():
    print('Test accuracy: %.2f%%' % accuracy(model,test_loader,device))


'(8) 数据可视化'
for batch_idx, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    '当遍历测试数据集中的第一个批次数据（即包含图像数据images、标签数据labels）后就使用 break语句 终止遍历'
    break

'images[0]表示 第一个批次数据 中的 第一个图像数据/特征图/3维tensor'
nhwc_image=np.transpose(images[0],axes=(1,2,0))
nhw_img = np.squeeze(nhwc_image.numpy(), axis=2)
plt.imshow(nhw_img, cmap='Greys')
plt.show()

'（9）查看 当前批次数据 中的 第一个图像数据/特征图/3维tensor 被模型预测为某个类别的概率 （以图像数据被模型预测为数字类别7为例） '
with torch.no_grad():
    images = images.to(device)
    logits, probability= model(images[0,batch_size])
    print('Probability 7 %.2f%% :' % (probability[0][7]*100))



