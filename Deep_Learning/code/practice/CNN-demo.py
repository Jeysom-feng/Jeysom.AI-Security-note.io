import torch.nn as nn
import matplotlib.pyplot as plt
import torch

def demo1():
    img =plt.imread(r"D:\Desktop\Python_file\DL_Project\data\cat.jpg")
    #print(img,img.shape)
    img1 =torch.tensor(img,dtype=torch.float32)
    img2 = img1.permute(2,0,1)  #将通道维度放到最前面
    #print(img2,img2.shape)
    img3 =img2.unsqueeze(0)  #在最前面增加一个维度
    #print(img3,img3.shape)

    #定义卷积层
    conv =nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,stride=1,padding=0)
    img4 =conv(img3)
    print(img4,img4.shape)
    img6 = img4[0]
    #print(img6,img6.shape)
    img7 =img6.permute(1,2,0)  #将通道维度放到最后面
    #print(img7,img7.shape)
    feature3 =img7[:,:,3].detach().numpy()  #将张量转换为numpy数组
    plt.imshow(feature3)
    plt.show()
if __name__ == '__main__':
    demo1()