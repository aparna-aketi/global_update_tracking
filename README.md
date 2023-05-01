# Global Update Tracking
This code is related to the paper titled, "Global Update Tracking: A Decentralized Learning Algorithm for Heterogeneous Data"

### Abstract
Decentralized learning enables the training of deep learning models over large distributed datasets generated at different locations, without the need for a central server. However, in practical scenarios, the data distribution across these devices can be significantly different, leading to a degradation in model performance. In this paper, we focus on designing a decentralized learning algorithm that is less susceptible to variations in data distribution across devices. We propose Global Update Tracking (GUT), a novel tracking-based method that aims to mitigate the impact of heterogeneous data in decentralized learning without introducing any communication overhead. We demonstrate the effectiveness of the proposed technique through an exhaustive set of experiments on various Computer Vision datasets (CIFAR-10, CIFAR-100, Fashion MNIST, and ImageNette), model architectures, and network topologies. Our experiments show that the proposed method achieves state-of-the-art performance for decentralized learning on heterogeneous data via a $1-6\%$ improvement in test accuracy compared to other existing techniques. 

# Available Models
* ResNet
* VGG-11
* MobileNet-V2
* LeNet-5

# Available Datasets
* CIFAR-10
* CIFAR-100
* Fashion MNIST
* Imagenette

# Available Graph Topologies
* Ring Graph
* Petersen Graph
* Dyck Graph
* Torus Graph
* Fully Connected Graph

# Requirements
* found in env.yml file

# Hyper-parameters
* --world_size  = total number of agents
* --graph       = graph topology (default ring); options: [ring, dyck, petersen, torus, full]
* --neighbors   = number of neighbor per agent (default 2)
* --arch        = model to train
* --normtype    = type of normalization layer
* --dataset     = dataset to train; ; options: [cifar10, cifar100, fmnist, imagenette]
* --batch_size  = batch size for training (batch_size = batch_size per agent x world_size)
* --epochs      = total number of training epochs
* --lr          = learning rate
* --momentum    = momentum coefficient
* --gamma       = averaging rate for gossip 
* --skew        = amount of skew in the data distribution (alpha of Dirichlet distribution); 0.01 = completely non-iid and 10 = more towards iid
* --scaling     = \textit{GUT} scaling factor $\mu$

# How to run?

test file contains the commands to run the proposed algorithms and baselines on various datasets, models and graphs
```
sh test.sh
```

Some sample commands:

ResNet-20 with 16 agents ring topology with GUT optimizer:
```
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=200 --arch=resnet --graph=ring --neighbors=2 --momentum=0.0 --scaling=0.9 --dataset=cifar10 --classes=10 --devices=4 --seed=12

```

ResNet-20 with 16 agents ring topology with QG-GUTm optimizer:
```
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=200 --arch=resnet --graph=ring --neighbors=2 --momentum=0.9 --scaling=0.06 --dataset=cifar10 --classes=10 --devices=4 --seed=12

```