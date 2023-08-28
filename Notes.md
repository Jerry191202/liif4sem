# LIIF4SEM Notes

## 1. 主要新增和修改

修改自原版LIIF（main分支），新增或的部分主要有：

1. `models/conv.py`：简化后仅由CNN构成的Encoder，取代原版的EDSR/RDN。
2. `uncertainty.py`：当网络中有Dropout时，评估模型的不确定性，具体做法是重复多次预测并计算输出图像每个像素点上的标准差。
3. `models/edsr.py`, `models/mlp.py`：修改了`EDSR`和`MLP`类，加入了Dropout层，可在训练参数配置文件中设置是否激活，以及设定dropout rate。
4. `datasets/image_folder.py`：修改了`ImageFolder`类，将原版的RGB图像（三通道）读取替换为灰度图像（单通道）读取。
5. `datasets/wrappers.py`：修改了`SRImplicitDownsampled`类，增加了向图像加入高斯噪声的代码（`line 141-144`），可在训练参数配置文件中设置是否激活。

## 2. 训练参数配置更新

训练参数配置文件主要以`configs/train-sem/`中的两个配置文件为模板，其中`train_convseq-liif.yaml`使用简化后仅由CNN构成的Encoder（`models/conv.py`），`train_edsr-baseline-liif.yaml`使用EDSR（`models/edsr.py`）作为Encoder。以`train_convseq-liif.yaml`为例，与原版配置文件相比新增之处主要有：

1. `line 15`, `line 32`：是否启用高斯噪声的选项；
2. `line 45`：Encoder中每一层CNN的channel数（即feature数），层数可变；
3. `line 47`：Encoder中CNN层的dropout rate，若为空则不启用Dropout；
4. `line 53`：ImNet中MLP层的dropout rate，若为空则不启用Dropout。

## 3. 训练数据和历史结果

数据已上传到[这里](https://we.tl/t-VTia4hFbtA)，下载后将load目录复制到`train_liif.py`同级目录下即可。

已有的训练结果存放在`save/`目录下，可使用tensorboard查看。

除了前面提到的文件以外，未使用到的多数原版代码文件都保留未删去，以防后续优化方法时需要再次使用。