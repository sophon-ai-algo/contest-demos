[TOC]

# 算能-人群计数模型迁移DEMO

本文以开源CSRNet为例，向选手介绍本次竞赛题目完整的实现流程。

其中model_tracing.py，data_handling.py仅为示例演示，并非固定使用，选手可自行更改，pretrained_model中提供的预训练模型源码仓库如下：

MCNN: https://github.com/svishwa/crowdcount-mcnn #(PyTotch)(unofficial)
     		 https://github.com/aditya-vora/crowd_counting_tensorflow #(TensorFlow)(unofficial)
CSRNet: https://github.com/leeyeehoo/CSRNet-pytorch
VGG: https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/vgg.py
CANNet: https://github.com/weizheliu/Context-Aware-Crowd-Counting

## 1 搭建本地开发环境

### 1.1 本地开发环境需求

- 开发主机：一台安装了Ubuntu18.04的x86架构下的64位系统主机，运行内存8GB以上(小于8G可能会在量化时中断）。
- 算能 docker镜像
- 算能 SDK2.7.0

### 1.2 安装docker

参考[《官方教程》](https://docs.docker.com/get-docker/)，若已经安装请跳过

```
#安装docker
sudo apt-get install docker.io
 
#创建docker用户组，之后docker命令可免root权限执行。（若已有docker组可忽略）
sudo groupadd docker 

# 将当前用户加入docker组 
sudo gpasswd -a ${USER} docker 

# 重启docker服务 
sudo service docker restart 

# 切换当前会话到新group或重新登录重启X会话 
newgrp docker 
```

### 1.3 获取docker镜像与SDK

点击前往[算能官网](https://sophon.cn/site/index.html)获取docker镜像下载链接

```
#通过wget命令下载docker镜像
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/03/19/13/bmnnsdk2-bm1684-ubuntu-docker-py37.zip 

#解压
unzip bmnnsdk2-bm1684-ubuntu-docker-py37.zip
```

点击前往[算能官网](https://sophon.cn/site/index.html)获取算能SDK2.7.0下载链接

```
#通过wget命令下载SDK2.7.0
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/08/15/09/bmnnsdk2_bm1684_v2.7.0_20220810patched.zip

#解压
unzip bmnnsdk2_bm1684_v2.7.0_20220810patched.zip
```

### 1.4 配置开发环境

```
#加载docker镜像
cd bmnnsdk2-bm1684-ubuntu-docker-py37/
docker load -i bmnnsdk2-bm1684-ubuntu.docker 

#解压缩SDK
tar zxvf bmnnsdk2-bm1684_v2.7.0.tar.gz 

#创建docker容器，SDK将被挂载映射到容器内部供使用
#若您没有执行前述关于docker命令免root执行的配置操作，需在命令前添加sudo
cd bmnnsdk2-bm1684_v2.7.0
./docker_run_bmnnsdk.sh
```

此时，已经进入docker容器中，接下来安装库

```
# 进入容器中执行自动安装库的脚本 
cd /workspace/scripts/ 
./install_lib.sh nntc 

# 配置环境变量，这一步会安装一些依赖库，并导出环境变量到当前终端 
# 导出的环境变量只对当前终端有效，每次进入容器都需要重新执行一遍此步骤
source envsetup_cmodel.sh
```

至此，开发环境已经配置完成，可以开始模型迁移啦！



## 2 模型转换

首先选用一个开源模型，将预训练的模型转换成可以在算能TPU上运行的bmodel形式，本文以CSRNet网络为例。

### 2.1 准备模型

这里需要首先明确选择的模型是否需要 trace，CSRNet 使用的是 pytorch 框架，需要通过 trace将.pth 转换为.pt。

#### i. 获取本文DEMO的工程源码 （trace 的时候需要用到源码的 misc 模块）

```
#下载git源码
git clone https://github.com/sophon-ai-algo/contest-demos.git
```

#### ii. 拷贝选择的预训练模型 

```
#将CSRNet的pth文件添加到pytorch_model目录下
cd NWPU-Crowd-Sample-Code/ 
mkdir pytorch_model
cd ..
cp ./pretrained_model/CSRNet-all_ep_529_mae_104.9_mse_433.5_nae_1.255.pth ./NWPU-Crowd-Sample-Code/pytorch_model
```

#### iii. 加入 model_tracing.py 文件，用于生成 traced_model.pt

```
#将 model_tracing.py 添加到 NWPU-Crowd-Sample-Code 目录下 (该 py 文件要和 misc 在同一目录，否则无法引包) 
python3 model_tracing.py --input-shape '(1,3,576,768)' \
--weights-path pytorch_model/CSRNet-all_ep_529_mae_104.9_mse_433.5_nae_1.255.pth \
--out-path pytorch_model/traced_model.pt

#如出现报错提示缺少easydict，则通过以下命令安装
pip3 install easydict
```

至此在pytorch_model 文件夹生成了 tarced_model.pt，则表示trace成功，模型准备完毕。

------

注：对于其他需要trace的模型，选手需要自己实现model_trace的过程。本文trace实现主要分为以下几个步骤，仅供参考。

```
a.修改NWPU-Crowd-Sample-Code中的对CSRNet的实现，将模型运行在cpu，并且用test_forward()函数替forward()函数
b.加载模型原始模型参数
c.用随机数trace pytorch模型
d.保存trace后的模型用于后面编译和量化（模型保存后缀为.pt）
```

model_tracing.py 四个参数，如下表所示，若使用其他模型需注意更改 model_tracing.py 内部参数。

本例中，input shape 在 NWPU-Cowd-Sample-Code/ saved_exp_para/C SRNet/NWPU.py 中可以找到。

| --model-name  | 默认为CSRNet              |
| ------------- | :------------------------ |
| --input-shape | 一般在源码里找输入的shape |
| --weight-path | 原始.pth文件路径          |
| --out-path    | 生成的.pt文件路径         |

### 2.2 编译模型： pt -> fp32bmodel

 参考算能SDK官方文档 https://sophgo-doc.gitbook.io/bmnnsdk2-bm1684/model-convert/fp32-bmodel/pt

#### i. 安装针对 pytorch 模型的编译工具 bmnetp

```
cd /bmnet/bmnetp
pip3 install bmnetp-2.7.0-py2.py3-none-any.whl
```

#### ii. 用 bmnetp 将 traced_model.pt 编译成 fp32bmodel

```
python3 -m bmnetp \
--model=./pytorch_model/traced_model.pt \
--shapes=[1,3,576,768] --net_name=CSRNet \
--outdir=./bmodel_fp32/ \
--target=BM1684
```

此处参数shape 与上面 input-shape 相同，执行后将生成 bmodel_fp32 文件夹，存放生成的 fp32bmodel文件。

### 2.3 模型量化： pt -> int8bmodel

int8 量化模型需要 lmdb 数据集。(auto_cali 的输入也可以是原始图片，详见[算能SDK的量化教程]( https://doc.sophgo.com/docs/2.7.0/docs_lat est_release/calibration-tools/html/module/chapter4.html#auto-cali)）

#### i.制作lmdb数据集

注 gen_lmdb.py 用来生成 lmdb 数据集，以.mdb 文件形式存储。gen_lmdb.py 的执行需要依托 misc, 所以量化数据集这一步需要在源码下进行。

将 gen_lmdb.py 添加到 NWPU-Crowd-Sample-Code 目录下（因为该.py 文件执行也需要引 misc）

将量化数据集添加到 NWPU-Crowd-Sample-Code 目录下（目前量化数据集 lmdb_dataSet 取的是训练数据集 NWPU -Crowd 数据集的前 200 张

将 lmdbdataSet.txt 文件添加到 NWPU-Crowd-Sample-Code 目录下（这个 txt 文件主要是为了获取 img_id 从而读取图片）

```
python3 gen_lmdb.py \
--img-path ./NWPU-Crowd \
--txt-path ./lmdbdataSet.txt \
--out-path ./lmdb
```

gen_lmdb.py 的三个参数

| --img-path | 原始图片数据目录            |
| ---------- | --------------------------- |
| --txt-path | 这里主要为了获取图片的index |
| --out-path | .mdb文件输出的目录          |

#### ii.一键量化命令 auto_cali。 

参考[量化工具使用说明]( https://doc.sophgo.com/docs/2.7.0/docs_latest_release/calibration-tools/html/index.html)

```
python3 -m ufw.cali.cali_model \
--net_name 'CSRNet' \
--model ./pytorch_model/traced_model.pt \
--cali_lmdb ./lmdb \
--cali_iterations 200 \
--input_shapes '(1,3,576,768)'
```

cali_iterations: 量化校准图片数量，默认值为 0 表示从 lmdb 或图片目录中获取真实数量,可以设置小于实际数量的图片数， 一般推荐使用 200 张图片校准

#### iii. 对比精度

bmrt_test用于测试精度，验证生成的 int8bmodel 是否正确。参考 [量化工具使用说明](https://doc.sophgo.com/docs/2.7.0/docs_latest_release/nntc/html/usage/bmrt_test.html)

```
#执行bmrt_test
bmrt_test --context_dir CSRNet
```



## 3 模型推理

推理部分均在算能云开发空间内完成，云空间的使用与环境配置请参考《CCFBDCI-算能竞赛云空间使用说明》

### 3.1 准备工作

将本地编译后的模型、测试集及所需相关文件上传至云空间：

------

- 本地 fp32bmodel 地址：/contest_demos/SOPHGO-CCF-Contest/NWPU-Crowd-Sample-Code/bmodel_fp32/ 

- 本地 int8bmodel 地址：/contest_demos/SOPHGO-CCF-Contest/NWPU-Crowd-Sample-Code/pytorch_model/CSRNet/ 

- 测试集地址：可从竞赛官网处下载获取 

- 相关文件：/contest_demo/SOPHGO-CCF-Contest/data_handling.py 

  ​                   /contest_demo/SOPHGO-CCF-Contest/TestDataSet.txt 

  ​                   /contest_demo/SOPHGO-CCF-Contest/test_sail.py

------

先对测试集原始数据尺寸进行统一处理。

具体代码实现在 data_handling.py，data_handling.py 的参数分别为--img-path和--out-path，分别表示输入数据集路径和输出数据集路径。

```
#对该数据集进行处理
python3 data_handling.py --img-path ./Test --out-path ./TestDataSet_handled 
```

执行后会生成 TestDataSet_handled 文件夹。

### 3.2 使用sail加载bmodel进行推理

选手可选择使用fp32或int8模式，最终只选择一种推理结果进行提交，推理结果命名为target.txt

加载fp32bmodel

```
python3 test_sail.py \
--data TestDataSet_handled \
--model bmodel_fp32/compilation.bmodel \
--result ./target.txt
```

加载int8bmodel

```
python3 test_sail.py \
--data TestDataSet_handled \
--model pytorch_model/CSRNet/compilation.bmodel \
--result ./target.txt
```

------

test_sail.py 的三个参数（这里要注意的是，当测试图片的命名方式不同时，要及时更改 test_all()函数，这里选择了上传TestDataSet.txt文件的方式获取img_id）

| --data   | 测试数据集目录 |
| -------- | -------------- |
| --model  | 选择的bmodel   |
| --result | 输出txt文件名  |

test_sail.py 注解：test_one()函数推理一张， test_all()遍历整个文件夹。执行这一步的时候如果报错缺少 sophon.sail，解决方法如下：

```
cd /data/bmnnsdk2/bmnnsdk2-bm1684_v2.7.0/scripts/
source envsetup_pcie.sh bmnetp
```

------

至此，你已经成功生成了target.txt文件，将该文件提交到CCFBDCI官方竞赛系统，看看自己的得分吧！
