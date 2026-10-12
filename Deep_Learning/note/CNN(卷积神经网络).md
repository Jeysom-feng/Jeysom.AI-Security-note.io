### 图像的基本知识：
#### 图像的基本概念：
图像时是由像素点构成的，每个像素点的取值范围为[0,255]，像素值越接近0，颜色越暗，接近于黑色，越接近255，颜色越亮，接近于白色



#### 图像分类：
##### 二值图：
一副二值图像的二维矩阵仅由0，1两个值组成，0代表黑色，1代表白色

##### 灰度图：
灰度图像矩阵元素的取值范围通常为[0,255]；因此其数据类型一般为8位无符号正数的（int8）,0代表纯黑色，255代表纯白色，中间的值从小到大代表从黑到白的过渡

##### 索引图：
索引图时单通道二维图，每个像素值代表的时类别编号

##### 真彩图：
真彩图时三通道二维图，与索引图一样可以表示彩色图像；

它由红，绿，蓝三种颜色的组合来表示每个像素的颜色，与索引图不同的是RGB图像每个像素的颜色值直接存放再图像矩阵中



### 卷积层：
#### 卷积层的计算方法：
用卷积核对图像的像素矩阵做哈达玛积，处理三通道RGB图时，卷积核是三维的，每个神经元有其各自的卷积核，卷积计算完后将三个特征图进行相加

#### 填充(Padding):
##### 作用：
###### 保持空间维度：
经过卷积后的特征图尺寸会缩小，多次卷积后，特征图会变得非常小，填充可以帮助维持输出特征图的尺寸与输入相同或接近相同

###### 保留边缘信息：
在卷积过程中，边缘像素参与的计算次数较少，容易丢失边缘信息，填充可以通过在边缘添加额外像素，增加边缘像素的参与度，从而更好地保留边缘信息

###### 提高性能：
Padding能很好的避免特征图快速缩小导致的信息丢失，从而提升模型性能，尤其是在较小图像进行多层卷积计算时



#### 步长(Stride):
步长是指卷积核在图像上滑动时的步伐大小，步长的大小影响着特征图的代大小，步长大可以提高每个神经元在输入数据时的感受野，意味着每个神经元可以捕捉到更大范围的输入信息，步长小可以保留更多的空间细节



##### 特征图大小的计算：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791196593464-faf72e14-2972-4c28-b3b1-39b9e0146f41.png" width="248.21427585113656" alt="" title="" crop="0,0,1,1" id="u155de7c6" class="ne-image">

#### 卷积层API示例：
```python
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

def demo1():
    img =plt.imread(r"D:\Desktop\Python_file\DL_Project\cat.jpg")
    #print(img,img.shape)
    img1 =torch.tensor(img,dtype=torch.float32)
    img2 = img1.permute(2,0,1)  #将通道维度放到最前面
    #print(img2,img2.shape)
    img3 =img2.unsqueeze(0)  #在最前面增加一个维度
    #print(img3,img3.shape)

    #定义卷积层
    conv =nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,stride=1,padding=1)
    img4 =conv(img3)
    #print(img4,img4.shape)
    img6 = img4[0]
    #print(img6,img6.shape)
    img7 =img6.permute(1,2,0)  #将通道维度放到最后面
    print(img7,img7.shape)
    feature3 =img7[:,:,3].detach().numpy()  #将张量转换为numpy数组
    plt.imshow(feature3)
    plt.show()
if __name__ == '__main__':
    demo1()
```



### 池化层：
池化是为了提取特征图中的重要特征，降低计算复杂度

池化后，输入通道数等于输出的通道数

##### 最大池化：
用k × k的矩阵在特征图中从上到下，从左右到右进行滑动，找出k × k矩阵中最大的数

##### 平均池化：
用k × k的矩阵在特征图中从上到下，从左右到右进行滑动，取k × k矩阵中数的平均值



池化层API:

最大池化：nn.Maxpool2d(kernel_size , stide , padding)

平均池化：nn.Avgpool2d(kernel_size , stide , padding)
