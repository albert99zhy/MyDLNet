import os
from datetime import datetime

from net import MyNet
from data import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import torch


DEVICE='cuda'


class Train:
    def __init__(self, root, weight_path):
        self.SummaryWriter = SummaryWriter('logs')  # 可视化 初始化
        self.train_dataset = MyDataset(root=root, is_train=True)  # 实例化训练数据集
        self.test_dataset = MyDataset(root=root, is_train=False)  # 实例化测试数据集
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=50, shuffle=True)  # 加载训练数据集
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=50, shuffle=True)  # 加载测试数据集
        self.net = MyNet().to(DEVICE)  # 实例化网络模型，放在显卡上
        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
        self.opt = optim.Adam(self.net.parameters())  # 构建优化器
        self.label_loss_fun = nn.BCEWithLogitsLoss()  # 交叉熵损失
        self.position_loss_fun = nn.MSELoss()  #
        self.sort_loss_fun = nn.CrossEntropyLoss()
        self.train = True
        self.test = True

    def __call__(self):
        index1, index2 = 0, 0
        for epoch in range(300):
            if self.train:
                for i, (img, label, position, sort) in enumerate(self.train_dataloader):
                    self.net.train()  # 可以省略
                    img, label, position, sort = img.to(DEVICE), label.to(DEVICE), position.to(DEVICE), sort.to(DEVICE)
                    out_label, out_position, out_sort = self.net(img)  # 生成网络模型输出
                    # print(out_label, out_position, out_sort)
                    '''调用各自损失函数计算相应损失'''
                    label_loss = self.label_loss_fun(out_label, label)
                    position_loss = self.position_loss_fun(out_position, position)
                    sort = sort[torch.where(sort >= 0)]
                    out_sort = out_sort[torch.where(sort >= 0)]
                    sort_loss = self.sort_loss_fun(out_sort, sort)
                    # print(label_loss, position_loss, sort_loss)
                    '''定义总损失函数'''
                    train_loss = label_loss + position_loss + sort_loss
                    '''梯度清零，反向传播，梯度更新'''
                    self.opt.zero_grad()
                    train_loss.backward()
                    self.opt.step()

                    if i % 10 == 0:
                        print(f'train_loss{i}=', train_loss.item())
                        self.SummaryWriter.add_scalar('train_loss', train_loss, index1)
                        index1 += 1

                date_time = str(datetime.now()).replace(' ', '-').replace('.', '_').replace(':', '_')
                torch.save(self.net.state_dict(), f'param/{date_time}-{epoch}.pt')

            if self.test:
                sum_sort_acc, sum_label_acc = 0, 0
                for i, (img, label, position, sort) in enumerate(self.train_dataloader):
                    # self.net.train()  # 可以省略
                    img, label, position, sort = img.to(DEVICE), label.to(DEVICE), position.to(DEVICE), sort.to(
                        DEVICE)
                    out_label, out_position, out_sort = self.net(img)  # 生成网络模型输出
                    # print(out_label, out_position, out_sort)
                    '''调用各自损失函数计算相应损失'''
                    label_loss = self.label_loss_fun(out_label, label)
                    position_loss = self.position_loss_fun(out_position, position)
                    sort = sort[torch.where(sort >= 0)]
                    out_sort = out_sort[torch.where(sort >= 0)]
                    sort_loss = self.sort_loss_fun(out_sort, sort)
                    # print(label_loss, position_loss, sort_loss)
                    '''定义总损失函数'''
                    test_loss = label_loss + position_loss + sort_loss
                    '''定义类'''
                    out_label = torch.tensor(torch.sigmoid(out_label))
                    out_label[torch.where(out_label >= 0.5)] = 1
                    out_label[torch.where(out_label < 0.5)] = 0
                    label_acc = torch.mean(torch.eq(out_label, label).float())
                    sum_label_acc += label_acc
                    '''计算sort精度'''
                    out_sort = torch.argmax(torch.softmax(out_sort, dim=1))
                    sort_acc = torch.mean(torch.eq(out_sort, sort).float())
                    sum_sort_acc += sort_acc

                    if i % 10 == 0:
                        print(f'test_loss{i}=', test_loss.item())
                        self.SummaryWriter.add_scalar('test_loss', test_loss, index2)
                        index2 += 1
                avg_sort_acc = sum_sort_acc / i
                avg_label_acc = sum_label_acc / i
                print(f'avg_sort_acc{epoch}=', avg_sort_acc)
                print(f'avg_label_acc{epoch}=', avg_label_acc)
                self.SummaryWriter.add_scalar('avg_sort_acc', avg_sort_acc, epoch)
                self.SummaryWriter.add_scalar('avg_label_acc', avg_label_acc, epoch)




if __name__ == '__main__':
    train = Train('D:/ZHY/MyDLnet/yellow_data', 'param/2023-04-08-10-18-11_335338-0.pt')
    train()
