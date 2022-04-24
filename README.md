# CycleMLP: A MLP-like Architecture for Dense Prediction

## 目录

* [1. 简介](#1-简介)
* [2. 数据集和复现精度](#2-数据集和复现精度)
* [3. 准备数据与环境](#3-准备数据与环境)
   * [3.1 准备环境](#31-准备环境)
   * [3.2 准备数据](#32-准备数据)
* [4. 开始使用](#4-开始使用)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型评估](#42-模型评估)
   * [4.3 模型预测](#43-模型预测)
   * [4.4 模型导出](#44-模型导出)
* [5. 代码结构](#5-代码结构)
* [6. 自动化测试脚本](#6-自动化测试脚本)
* [7. License](#7-license)
* [8. 参考链接与文献](#8-参考链接与文献)

## 1. 简介

这是一个PaddlePaddle实现的CycleMLP。

<p align="middle">
  <img src="https://github.com/ShoufaChen/CycleMLP/blob/main/figures/teaser.png?raw=true" height="300" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/ShoufaChen/CycleMLP/blob/main/figures/flops.png?raw=true" height="300" />
</p>

**论文:** [CycleMLP: A MLP-like Architecture for Dense Prediction](https://arxiv.org/abs/2107.10224)

**参考repo:** [CycleMLP](https://github.com/ShoufaChen/CycleMLP)

项目aistudio地址：

notebook任务：https://aistudio.baidu.com/aistudio/projectdetail/3877397

在此非常感谢`ShoufaChen`贡献的[CycleMLP](https://github.com/ShoufaChen/CycleMLP)，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

数据集为ImageNet，训练集包含1281167张图像，验证集包含50000张图像。

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

您可以从[ImageNet 官网](https://image-net.org/)申请下载数据。

| 模型      | top1 acc (参考精度) | top1 acc (复现精度) | 权重 \| 训练日志 |
|:---------:|:------:|:----------:|:----------:|
| CycleMLP-B1 | 0.789 | 0.790 | checkpoint-best.pd \| train.log |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1D75OdwxWOxf9RWnzvD2ixg?pwd=oduk)


## 3. 准备数据与环境


### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * RTX3090
- 框架：
  - PaddlePaddle >= 2.2.0

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 下载代码

```bash
git clone https://github.com/flytocc/CycleMLP-paddle.git
cd CycleMLP-paddle
```

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

如果您已经ImageNet1k数据集，那么该步骤可以跳过，如果您没有，则可以从[ImageNet官网](https://image-net.org/download.php)申请下载。


## 4. 开始使用


### 4.1 模型训练

* 单机多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    main.py \
    --model=CycleMLP_B1 \
    --batch_size=256 \
    --data_path=/path/to/imagenet/ \
    --output_dir=./output/ \
    --dist_eval
```

部分训练日志如下所示。

```
[16:56:29.233819] Epoch: [261]  [ 920/1251]  eta: 0:05:50  lr: 0.000052  loss: 3.4592 (3.3812)  time: 1.0303  data: 0.0012
[16:56:49.578909] Epoch: [261]  [ 940/1251]  eta: 0:05:29  lr: 0.000052  loss: 3.7399 (3.3853)  time: 1.0171  data: 0.0015
```

### 4.2 模型评估

``` shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    eval.py \
    --model=CycleMLP_B1 \
    --batch_size=256 \
    --data_path=/path/to/imagenet/ \
    --dist_eval \
    --resume=$TRAINED_MODEL
```

### 4.3 模型预测

```shell
python infer.py \
    --model=CycleMLP_B1 \
    --infer_imgs=./demo/ILSVRC2012_val_00020010.JPEG \
    --resume=$TRAINED_MODEL
```

<div align="center">
    <img src="./demo/ILSVRC2012_val_00020010.JPEG" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 211, 210, 209, 236], 'scores': [0.8659181594848633, 0.004747727885842323, 0.003118610242381692, 0.0025468438398092985, 0.0017893684562295675], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'German short-haired pointer', 'Chesapeake Bay retriever', 'Doberman, Doberman pinscher']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.8659181594848633`。

### 4.4 模型导出

```shell
python export_model.py \
    --model=CycleMLP_B1 \
    --output_dir=./output/ \
    --resume=$TRAINED_MODEL
```

## 5. 代码结构

```
├── cycle_mlp.py
├── demo
├── engine.py
├── eval.py
├── export_model.py
├── infer.py
├── main.py
├── README.md
├── requirements.txt
├── test_tipc
└── util
```


## 6. 自动化测试脚本

**详细日志在test_tipc/output**

TIPC: [TIPC: test_tipc/README.md](./test_tipc/README.md)

首先安装auto_log，需要进行安装，安装方式如下：
auto_log的详细介绍参考https://github.com/LDOUBLEV/AutoLog。
```shell
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```
进行TIPC：
```bash
bash test_tipc/prepare.sh test_tipc/config/CycleMLP/CycleMLP_B1.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/config/CycleMLP/CycleMLP_B1.txt 'lite_train_lite_infer'
```
TIPC结果：

如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到`test_tipc/output/`文件夹中的文件中。

```bash
Run successfully with command - python3.7 eval.py --model=CycleMLP_B1 --data_path=./dataset/ILSVRC2012/ --cls_label_path=./dataset/ILSVRC2012/val_list.txt --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/CycleMLP_B1/checkpoint-latest.pd !
Run successfully with command - python3.7 export_model.py --model=CycleMLP_B1 --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/CycleMLP_B1/checkpoint-latest.pd --output=./test_tipc/output/norm_train_gpus_0_autocast_null !
Run successfully with command - python3.7 inference.py --use_gpu=True --use_tensorrt=False --precision=fp32 --model_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdmodel --batch_size=2 --input_file=./dataset/ILSVRC2012/val  --params_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdiparams > ./test_tipc/output/python_infer_gpu_usetrt_False_precision_fp32_batchsize_2.log 2>&1 !
...
```

* 更多详细内容，请参考：[TIPC测试文档](./test_tipc/README.md)。


## 7. License

CycleMLP is released under MIT License.


## 8. 参考链接与文献
1. CycleMLP: A MLP-like Architecture for Dense Prediction: https://arxiv.org/abs/2107.10224
2. CycleMLP: https://github.com/ShoufaChen/CycleMLP
