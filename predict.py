import torch
import cv2
from net import MyNet
import os


if __name__ == '__main__':
    img_name = os.listdir(r'D:/ZHY/MyDLnet/yellow_data/test')
    for i in img_name:
        img_dir = os.path.join('D:/ZHY/MyDLnet/yellow_data/test', i)
        img = cv2.imread(img_dir)
        print(i.split('.')[2:6])  # 读坐标值
        sort = i.split('.')[6]
        position = i.split('.')[2:6]
        position = [int(j) for j in position]
        cv2.rectangle(img, (position[0], position[1]), (position[2], position[3]), (0, 255, 0), thickness=2)
        cv2.putText(img, sort, (position[0], position[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=2)

        model = MyNet()
        model.load_state_dict(torch.load('param/2023-04-08-10_35_45_607516-10.pt'))
        new_img = torch.tensor(img).permute(2, 0, 1)
        new_img = torch.unsqueeze(new_img, dim=0) / 255
        out_label, out_position, out_sort = model(new_img)

        out_label = torch.sigmoid(out_label)
        out_sort = torch.argmax(torch.softmax(out_sort, dim=1))

        out_position = out_position[0] * 300
        out_position = [int(i) for i in out_position]
        if out_label > 0.5:
            cv2.rectangle(img, (out_position[0], out_position[1]), (out_position[2], out_position[3]), (0, 0, 255),
                          thickness=2)
            cv2.putText(img, str(out_sort.max().item()), (out_position[0], out_position[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), thickness=2)

        cv2.imshow('img', img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

