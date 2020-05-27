# ddl-benchmarks: Benchmarks for Distributed Deep Learning.

## Introduction 
This repository contains a set of benchmarking scripts for evaluating the training performance of popular distributed deep learning methods, which mainly focuses on system-level optimization algorithms of synchronized stochastic gradient descent with data parallelism. Currently, it covers:
### system architectures
- Parameter server with [BytePS](https://github.com/bytedance/byteps).
- All-to-all with [Horovod](https://github.com/horovod/horovod).
### optimization algorithms
- Wait-free backpropagation (WFBP), which is also known as the technique of pipelining the backward computations with gradient communications and it is a default feature in current deep learning frameworks.
- Tensor fusion, which has been integraded in Horovod with a hand-craft threshold to determine when to fusion tensors, but it is possible to dynamically determine to fusion tensors in [MG-WFBP](https://github.com/HKBU-HPML/MG-WFBP).
- Tensor partition and priority schedule, which are proposed in [ByteScheduler](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler).
- Gradient compression with quantization (i.e., [signSGD](https://github.com/jiaweizzhao/signSGD-with-Majority-Vote)) and sparsification (i.e., [TopK-SGD](https://github.com/hclhkbu/gtopkssgd)). These methods are included in the code, but they are excluded from our paper as the paper focuses on the system-level optimization methods.
### deep neural networks
- [Convolutional neural networks (CNNs)](https://pytorch.org/docs/stable/torchvision/models.html) on a fake ImageNet data set (i.e., randomly generate the input image of 224\*224\*3)
- [Transformers](https://github.com/huggingface/transformers): BERT-Base and BERT-Large pretraining models.

## Installation
### Prerequisites
- Python 3.6+
- CUDA-10.+
- NCCL-2.4.+
- [PyTorch-1.4.+](https://download.pytorch.org/whl/torch_stable.html)
- [OpenMPI-4.0.+](https://www.open-mpi.org/software/ompi/v4.0/)
- [Horovod-0.19.+](https://github.com/horovod/horovod)
- [BytePS-0.2.+](https://github.com/bytedance/byteps)
- [ByteScheduler](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler)
- [bit2byte](https://github.com/jiaweizzhao/signSGD-with-Majority-Vote/tree/master/main/bit2byte-extension): Optional if not run signSGD.

### Get the code
```
$git clone https://github.com/HKBU-HPML/ddl-benchmarks.git
$cd ddl-benchmarks 
$pip install -r requirements.txt
```

### Configure the cluster settings
Before running the scripts, please carefully configure the configuration files in the directory of `configs`.
- configs/cluster\*: configure the host files for MPI
- configs/envs.conf: configure the cluster enviroments

Create a log folder, e.g., 
```
$mkdir -p logs/pcie
```

### Run benchmarks
- The batch mode
```
$python benchmarks.py
```
- The individual mode, e.g.,
```
$cd horovod
$dnn=resnet50 bs=64 nworkers=64 ./horovod_mpi_cj.sh
```

## Paper
If you are using this repository for your paper, please cite our work
```
@article{shi2020ddlsurvey,
    author = {Shi, Shaohuai and Tang, Zhenheng and Chu, Xiaowen and Liu, Chengjian and Wang, Wei and Li, Bo},
    title = {Communication-Efficient Distributed Deep Learning: Survey, Evaluation, and Challenges},
    journal = {arXiv},
    year = {2020}
}
```
