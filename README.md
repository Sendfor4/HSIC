# HSIC
## 关于HybridSN
### 关于这个仓库和它的网络
自己的笔记本是没有搭载cuda的，所以只简单的训练了50个epoch，看了一下预测结果，但是即使是这样都运行了30分钟左右。<br>
阅读博客和代码的时候，只关注了预处理部分和网络的搭建，预测部分没怎么关注，是分别从这里<https://tanxy.club/2022/HybridSN>还有这里<https://github.com/Vaibhavdixit02/HybridSN.git>各自拿取了一部分，然后直接使用的。毕竟它这个高光谱的话，是对像素点进行预测，不是像普通的图像分类一样，是对整张图像进行预测。
####
HybridSN的论文地址:<https://arxiv.org/abs/1902.06701><br>
HybridSN的TensorFlow实现方式：<https://github.com/gokriznastic/HybridSN.git><br>
HybridSN的PyTorch的实现方式：<https://github.com/Pancakerr/HybridSN.git><br>
### 关于HybridSN_person.ipynb和HybridSN.ipynb
(1)HybridSN_person.ipynb文件是自己参考了多篇博客和官方论文以及一些开源代码后，直接使用仓库所指向的PyTorch实现方法中的构造，然后拼凑出来的demo，包含了训练过程和预测过程，预测图像都已经嵌到了notebook当中。生成了一个已经训练好的模型，一张所使用的数据集的RGB伪彩色图像，一张GT值图像，一份classification_report.txt分类报告。
最开始的时候网络结构和这里<https://tanxy.club/2022/HybridSN>是差不多的，但是自己在使用tensorboard进行模型可视化后大概能理解它这里<https://github.com/Pancakerr/HybridSN.git>为什么要这样搭建了，所以直接拿过来使用了。<br>

(2)HybridSN.ipynb就是这篇论文<https://arxiv.org/abs/1902.06701>所指向的仓库（<https://github.com/gokriznastic/HybridSN.git>，使用tensorflow实现）中，再指向的另一个仓库使用pytorch（<https://github.com/Pancakerr/HybridSN.git>，使用PyTorch实现）实现的方式，但是我清空了所有的输出。<br>
### 关于参考博客、论文、源码
论文地址：<https://arxiv.org/abs/1902.06701><br>
官方PyTorch实现方式：<https://github.com/Pancakerr/HybridSN><br>
官方TensorFlow实现方式：<https://github.com/gokriznastic/HybridSN><br>
参考博客：<https://tanxy.club/2022/HybridSN>，<https://blog.csdn.net/qduljy/article/details/112134529><br>
参考源码：<https://github.com/Vaibhavdixit02/HybridSN.git>，<https://github.com/whu-pzhang/HybridSN.git><br>
### 关于预测图像的画布大小问题
当设置figsize = (5,5)或者更小（比如figsize = (4,4)）时所呈现的预测图像是比较怪的，但是如果设置figsize = (6,6)或者figsize =(7,7)，预测图像都比较正常
### 关于GT值图像和RGB origin图像
同样是直接使用这篇论文中官方实现的代码，从而嵌入图像的。<br>

使用PU数据集所直接嵌入到notebook的GT值图像和保存到本地的GT值图像并不一样。而对于IP数据集，GT值图像无论是直接嵌入到notebook当中，还是保存到本地当中，所显示的GT值图像都是一样的。SA数据集并没有进行尝试<br>

如果使用demo进行训练，PU数据集显示的结果则更像嵌入到notebook当中的GT值图像。对于IP数据集，由于GT值图像无论是直接嵌入到notebook当中，还是保存到本地当中，所显示的GT值图像都是一样的，所以说更像哪一张都不过分。<br>
### 关于预处理
    目前见到的大致分为以下几种:
（1）对数据进行降维，将原来的数据集由原来的```H x W x C```变成```H x W x B```的情况，然后对每个像素点生成大小为```S x S x B```的3D-patch。<br>
（2）不对数据进行降维，但是对像素划分成```S x S x C```的3D-patch。<br>
（3）不做任何的预处理，直接使用，比如这一篇《Going Deeper with Contextual CNN for Hyperspectral Image Classification》<https://arxiv.org/pdf/1604.03519><br>
### 关于Classificationmodel.py
大概放置了我翻看别人提供的开源代码里，提供的一些比较容易看懂的模型架构<br>
#### 
对于HResNet，我暂时没有找到有关的论文或者文章，但是速度比较快，层结构也比较简单<br>

对于ResNet17和ResNet34，最初找到的是ResNet17，但是这个ResNet17一开始使用的是3x3的卷积核进行卷积，而不是像ResNet原论文（[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)）中一样，一开始使用7x7的卷积核进行卷积，而且在4个stage前后也没有使用池化层。而在ResNet原论文（[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)）中，ResNet17和ResNet34所描述的结构是一样的，所以在ResNet17的基础上简单修改了一下，多加了一个4个不同stage的重复次数（ResNet17中是[2,2,2,2]）会出现[3,4,6,3]的情况。<br>

关于FAST3DCNN，它的这篇论文中的架构和HybridSN中的架构很相似啊，具体的我就不过多叙述，可以翻阅两篇论文查看，都比较短。<br>

关于HybridSN，这里不再过多赘述<br>

关于SSRN，比较具体的部分我还在调整，我也不过多叙述<br>
#### HResNet、FAST3DCNN、HybirdSN都已经生成了可视化模型
关于模型可视化，请在Terminal中使用诸如```tensorboard --logdir C:\Users\81635\PyWORKSPACE\HSIC\logs\FAST3DCNN\.```的语句，指向日志所在的文件地址，请使用绝对路径，我不能保证不同的编辑器使用相对路径会不会出错<br>
####
ResNet：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)<br>
FAST3DCNN：<https://arxiv.org/pdf/2004.14152><br>
HybridSN：<https://arxiv.org/pdf/1902.06701><br>
SSRN：<https://www.researchgate.net/publication/320145356_Deep_Residual_Networks_for_Hyperspectral_Image_Classification>

### 关于自己的一些碎碎念
不得不说这几个星期有那么几天真的看源码看得我头疼，有时候白天醒了就在跑别人的demo，感觉不对又继续跑一次，有些demo甚至会因为对数据进行翻倍操作的情况，然后自己将这个patch设置大一点就让kaggle跑不动，直接警告说内存爆了。
####
谷歌colab免费GPU时间砍了很多了，经常前天晚上跑了两三个钟头，第二天再想用就直接说限额了，不给你GPU了。相比之下kaggle每个星期提供免费的30小时GPU加速和20小时TPU加速真的好很多。<br>
####
我不知道别人对开组会这件事怎么想啊…………，相比于其他老师手下的学生(起码我所见到的和认识的)，我们这边开的次数真的算是很多的了，而且也不怎么催，其他人甚至还有开学到现在都没见到导师的。而且每次开完之后，不止是我，好像其他人也多多少少会有点厌学情绪。
 
