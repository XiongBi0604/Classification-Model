'''
@author: Bi Xiong
@time: 2024/4/7 22:16
'''


'（1）导入所需的库'
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import  matplotlib.pyplot as plt


'（2） 参数设置'
'设置随机种子，使得随机操作（如数据集的划分等）在每次运行代码时都能拥有相同的随机结果'
random_seed = 1
lr = 0.001
epochs = 100

'这里是手动设置的batch_size值，感兴趣的朋友可以使用 .view() 方法来自动地计算出batch_size值'
batch_size = 100

'将分类数设置为10，根据自己的目标任务灵活设置'
num_classes = 10

'torch.cuda.is_available() 函数用于检查电脑系统中是否有可用的 CUDA 设备（GPU）'
'如果 torch.cuda.is_available() 返回 True，就选择字符串cuda（此时选用GPU设备）；若返回False, 则选择字符串cpu（此时选用CPU设备）'
'最后将上述条件表达式的值赋值给变量device，且device表示模型和tensor所在的设备'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'torch.backends.cudnn.deterministic = True 表明CUDA设备可用。这行代码设置了CuDNN后端为确定性模式，CuDNN是一个深度学习库，用于加速神经网络的训练'
'启用确定性模式可以使每次训练得到相同的结果，这对于实验的可重复性是很重要的'
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

'注意，上述两个代码块都是有必要使用的，不存在重复。变量device指定运行代码的设备，确定性模式确保在使用GPU加速训练时每次获得的实验结果相同'


'（3）数据集的准备'
'（3.1）定义数据预处理操作'
'使用transforms.Resize()函数将输入图像的尺寸调整为指定的大小，且该函数的输入实参是一个元组，用来指定输入图像的目标大小'
'使用transforms.ToTensor()函数将输入图像转换为Tensor，并将图像的像素值归一化到[0，1]之间，且该函数无输入实参，本身就只是一个转换函数'
'使用transforms.Normalize()函数对输入图像进行标准化，即对输入图像每个通道上的像素值先减去均值，再除以标准差。其中，该函数的输入实参是两个元组，第一个元组是均值元组，第二个元组是标准差元组'
'最后，使用transforms.Compose([])函数将各类的transforms函数按顺序打包为一个整体，然后作用于输入图像'
transforms = transforms.Compose([transforms.Resize((32,32)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])

'（3.2）加载数据集'
'定义训练集。root表示下载训练数据到目录名为data的文件中，transform表示对训练数据进行的数据预处理操作'
train_dataset = datasets.MNIST(root='data',train=True,download=True,transform=transforms)
'定义测试集。root表示下载测试数据到目录名为data的文件中，transform表示对测试数据进行的数据预处理操作'
test_dataset = datasets.MNIST(root='data',train=False,download=True, transform=transforms)

'（3.3）制作数据加载器'
'定义训练数据加载器。shuffle=True表示打乱训练数据的顺序，num_workers=4表示用4个子进程并行加载训练数据，提高数据加载的速度'
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
'定义测试数据加载器。shuffle=False表示不打乱测试数据的顺序，num_workers=4表示用4个子进程并行加载测试数据，提高数据加载的速度'
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)


'（4）构建分类模型LeNet'
'（4.1）定义网络结构'
class LeNet(nn.Module):

    '定义网络层（卷积层、池化层、全连接层）'
    def __init__(self, num_classes):
        super(LeNet, self).__init__()

        'nn.Sequential() 的用法与transforms.Compose([])类似，都相当于一个打包函数,只不过nn.Sequential()函数是用来按顺序打包网络层的（如卷积层、池化层、全连接层等）'
        self.feature_stage = nn.Sequential(

        #  因为MNIST数据集中的图像为灰度图像，所以这些图像的输入通道数为 1'
        #  输入形状（32，32，1）， 输出形状（28,28,6）
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),

        #  输入形状为（28，28，6），输出形状（14,14,6）
        nn.MaxPool2d(kernel_size=2, stride=2),

        #  输入形状为（14,14,6），输出形状为（10,10,16）
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),

        #  输入形状为（10,10,16），输出形状为（5,5,16）
        nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classification_stage = nn.Sequential(

            nn.Linear(in_features=16 * 5 * 5, out_features=120),

            nn.Linear(in_features=120, out_features=84),

            nn.Linear(in_features=84, out_features=num_classes)
        )

    '定义网络的前传（正向）传播'
    def forward(self, x):
        x = self.feature_stage(x)

        '使用 .view() 方法将前面卷积层/池化层的输出（实际为4维tensor，即（batch_size,C,H,W），但一般看作3维tensor（C,H,W））转换为全连接层的输入（实际为2维tensor（batch_size，in_features），但一般看作1维向量）'
        '其中in_features表示全连接层的输入特征数（或 特征图（即3维tensor）被展平为1维向量时所具有的长度。并且in_features = C×H×W'
        '例如，在这里in_features = C×H×W = 16×5×5'
        x = x.view(100, 16 * 5 * 5)

        '-----------------------------------------------------------------------------------'
        '注意：由于前面手动设置了batch_size的值，因此在这里就不需要使用占位符 -1 的方式来自动地求batch_size值，即不需要写成'
        'x = x.view(-1, 16 * 5 * 5)； 而是直接将 -1 改为手动设置的batch_size值'
        '----------------------------------------------------------------------------------'

        '当1维向量x（实际为2维tensor, 即（batch_size，in_features））输入最后一个全连接层后，得到模型的原始输出 logits'
        '至于为什么叫原始输出？因为logits未经激活函数softmax()处理'
        logits = self.classification_stage(x)

        '使用激活函数softmax()在logits的第二个维度（即dim=1）上将logits转换为各个类别的预测概率，即probability'
        probability = F.softmax(logits, dim=1)
        return logits, probability

'（4.2）实例化对象'
'调用设置随机种子的函数，用来指定随机种子的值'
torch.manual_seed(random_seed)

'实例化LeNet为model'
model = LeNet(num_classes)

'将model部署到设备device上'
model = model.to(device)

'选择优化器。其中model.parameters()表示模型的权重参数'
optimizer = torch.optim.Adam(model.parameters(),lr=lr)


'（5）定义准确率函数'
def accuracy(model, data_loader, device):

    'correct_pred, num_examples分别表示正确预测的数量、样本数量'
    correct_pred, num_examples = 0, 0

    'enumerate() 函数用于遍历集合中的元素及其索引（序号）'
    '遍历数据加载器，其中images、labels分别表示图像数据、标签数据'
    for i, (images, labels) in enumerate(data_loader):

        '将images、labels部署到设备device上'
        images = images.to(device)
        labels = labels.to(device)

        '将images输入到模型中，得到模型的原始输出logits 、各个类别的预测概率值probability'
        logits, probability = model(images)

        'torch.max()函数的第一个参数 probability给定了样本属于每个类别的预测概率；第二个参数 dim=1 指定了沿着第二个维度（列）对probability进行操作，即在每行中找最大预测概率值'
        'torch.max()函数返回一个二元组。第一个值是每行中的最大预测概率值。（而我们不需要第一个值，因此用 _ 表示） 第二个值是每行中最大预测概率值所对应的索引，即最大预测概率值所对应的具体类别。（我们用 predicted_labels 表示）'
        _, predicted_labels = torch.max(probability, dim=1)

        'predicted_labels == labels创建一个布尔张量。若对应位置上的元素其预测标签和真实标签相等时为 True，否则为 False。 .sum() 对上述布尔张量进行求和操作，得到 预测标签和真实标签相等的元素的总数量，即当前批次中正确预测的样本数量'
        correct_pred += (predicted_labels == labels).sum()

        'labels.size(0)返回 标签数据 labels 在第一个维度上的大小，即样本数量。这行代码的作用是将当前批次中的样本数量加到总样本数量（num-examples）中，以便在训练过程中跟踪已处理过的总样本数'
        num_examples += labels.size(0)
    return correct_pred / num_examples * 100


'（6）训练model'
'作为训练模型的时间起点'
start_time = time.time()

'遍历训练次数。其中epochs表示总的训练次数'
for epoch in range(epochs):

    '将模型设置为训练模式'
    model.train()

    '遍历训练数据加载器中的图像数据、标签数据、以及索引'
    'batch_idx 表示 当前批次索引（序号），即第几个批次'
    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        '前向传播'
        logits, probability = model(images)

        '计算损失。labels是真实标签，logits是原始输出，计算损失的时候是用logits、labels，没有probability（因为它是概率值，无法与真实标签进行比对，自然也算不了损失值）'
        'reduction=sum 表示对每个图像样本的损失进行求和，得到当前批次的总损失'
        loss = F.cross_entropy(logits, labels,reduction='sum')

        '梯度清零。因为PyTorch默认会累积梯度，而在加载每个新批次数据之前需要清空之前的梯度'
        optimizer.zero_grad()

        '反向传播。计算损失相对于权重参数的梯度（所以用loss来进行反向传播）'
        loss.backward()

        '更新权重参数'
        optimizer.step()

        '打印 当前批次索引 所对应的训练日志'
        '先判断 当前批次索引 是不是50的倍数，若是，就打印以下两个内容'
        if batch_idx % 50 == 0:
            print('Epoch: %03d | %03d   Batch: %03d | %03d   Loss：%.4f' % (
            epoch + 1, epochs, batch_idx, len(train_loader), loss))
            print('Train accuracy: %.2f%%' % accuracy(model, train_loader, device))

            '----------------------------------------------------------------------'
            '%03d表示输出格式为3位整数、 %.2f表示输出格式为2位小数、 %%表示用百分数表示'
            '----------------------------------------------------------------------'

    '计算训练集上 完成一次训练（即在一个epoch数下，遍历完所有的 batch_idx数 所花的时间'
    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

'计算训练集上 完成所有次训练（即遍历完所有的epoch数） 所花的时间'
print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))


'（7）测试model'
'这里也可以将 with torch.no_grad() 替换为 model.eval()'
'但是本人习惯使用with torch.no_grad() 或者 with torch.set_grad_enabled(False) ，因为它们和model = model.eval()有类似的效果，即（1）在测试集上进行评估时禁用梯度计算、（2）将模型设置为评估模式'
'但前两者更加灵活，可以包含更多操作，且不需要手动切换模型的状态。同时，前两者的区别为：前者主要用于在某些地方关闭梯度计算，而后者主要用于在整个代码块中全局关闭梯度计算'
with torch.no_grad():
    print('Test accuracy: %.2f%%' % accuracy(model,test_loader,device))


'(8) 数据可视化'
for batch_idx, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)

    '当遍历测试数据加载器中的第一个批次数据（包含图像数据images、标签数据labels）后就使用 break语句 终止遍历'
    break

'images[0]表示 第一个批次数据 中的 第一个图像数据（对应第一个3维tensor（C,H,W），且此时的维度索引为：C-0 , H-1 , W-2）'
'使用np.transpose()函数对图像数据进行转置，同时将图像数据的维度顺序进行变化，变为（H,W,C），即：H-1，W-2，C-0。因为要满足matplotlib库中图像的维度顺序，即（H, W, C）'
nhwc_image = np.transpose(images[0], axes=(1, 2, 0))

'对3维tensornhwc_image使用 .numpy() 方法使其变为numpy数组'
'使用np.squeeze()函数删除数组中维度为2的所有数据，即删除通道维度C上的所有数据'
nhw_img = np.squeeze(nhwc_image.numpy(), axis=2)

'调用plt.imshow()函数绘制图像，且cmap=Greys表示绘制的图像类型为灰度图像'
plt.imshow(nhw_img, cmap='Greys')

'将绘制的图像显示在屏幕上'
plt.show()

'（9）查看第一个图像数据被模型预测为某个类别的概率 （以图像数据被模型预测为数字类别7为例） '
with torch.no_grad():
    images = images.to(device)

    '0表示选择该批次数据中的第一个图像数据，同时在第一个图像数据的基础上再添加一个batch_size维度（即将3维tensor还原为实际的4维tensor），因为模型的输入通常是具有batch_size维度的4维tensor。'
    '如果是利用占位符-1的方式来自动求batch_size的值，那么就写成 images.[0, None] 的形式。如果是手动设置batch_size的值，那么就写成 images.[0, batch_size] 的形式。'
    logits, probability = model(images[0, batch_size])

    '0表示该批次数据中的第一个图像数据，7表示模型将第一个图像数据预测为数字类别7'
    print('Probability 7 %.2f%% :' % (probability[0][7] * 100))




