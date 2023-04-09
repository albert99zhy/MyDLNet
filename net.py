from torch import nn
import torch


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 11, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(11, 22, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(22, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
        )
        self.label_layer = nn.Sequential(  # 二分类 是否
            nn.Conv2d(128, 1, 19),
            nn.ReLU()
        )
        self.position_layer = nn.Sequential(  # 位置回归
            nn.Conv2d(128, 4, 19),
            nn.ReLU()
        )
        self.sort_layer = nn.Sequential(  # 多分类
            nn.Conv2d(128, 20, 19),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layers(x)
        label = self.label_layer(out)
        label = torch.squeeze(label, dim=2)
        label = torch.squeeze(label, dim=2)
        label = torch.squeeze(label, dim=1)
        position = self.position_layer(out)
        position= torch.squeeze(position, dim=2)
        position = torch.squeeze(position, dim=2)
        sort = self.sort_layer(out)
        sort = torch.squeeze(sort, dim=2)
        sort = torch.squeeze(sort, dim=2)
        return label, position, sort


if __name__ == '__main__':
    net = MyNet()
    x = torch.randn(3, 3, 300, 300)
    print(net(x)[2].shape)
