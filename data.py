import os
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import numpy as np


class MyDataset(Dataset):
    def __init__(self, root, is_train=True):
        # 导入数据，如果是图片一般保存图片的路径
        self.dataset = []
        dir = 'train' if is_train else 'test'
        sub_dir = os.path.join(root, dir)
        # print(sub_dir)
        img_list = os.listdir(sub_dir)
        for i in img_list:
            img_dir = os.path.join(sub_dir, i)
            # print(img_dir)
            self.dataset.append(img_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):  # 当数据被调用时就会出发该函数
        data = self.dataset[index]
        img = cv2.imread(data)/255
        # img = Image.open(data)  # 用此(PIL)图片读取+torch转 C W H失败，但是用此+np转C W H成功
        # new_img = np.transpose(img, (2, 0, 1))
        new_img = torch.tensor(img).permute(2, 0, 1)
        # print(new_img.shape)
        """
        是否包含物体， 用点分隔  
        这里这些信息包含在图片名中
        数据集图片名称中第0位是序号，第一位是是否包含，其余的是位置
        """
        data_list = data.split('.')
        # print(data_list)
        label = int(data_list[1])
        position = data_list[2:6]
        position = [int(i)/300 for i in position]
        '''分类(不同的小黄人)20类'''
        sort = int(data_list[6])-1
        return np.float32(new_img), np.float32(label), np.float32(position), np.int(sort)


if __name__ == '__main__':
    data = MyDataset('D:\ZHY\MyDLnet\yellow_data')
    for i in data:
        print(i)
