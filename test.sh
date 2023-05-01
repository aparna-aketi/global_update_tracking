##########################################################################################################################
# Commands to run experiments on 16-node ring, training CIFAR-10 with ResNet-20, alpha=0.1

## 1. DSGD -- momentum and scaling are set to 0
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=200 --arch=resnet --graph=ring --momentum=0.0 --scaling=0.0 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..

## 2. GUT -- momentum is set to 0 and scaling is set to 0.9
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=200 --arch=resnet --graph=ring --momentum=0.0 --scaling=0.9 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..

## 3. QG-DSGDm -- momentum is set to 0.9 and scaling is set to 0
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=200 --arch=resnet --graph=ring --momentum=0.9 --scaling=0.0 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..

## 4. QG-GUTm -- momentum is set to 0.9 and scaling is set to 0.06
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=200 --arch=resnet --graph=ring --momentum=0.9 --scaling=0.06 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..
##########################################################################################################################
# Commands to run experiments on 32-node ring, training CIFAR-10 with VGG-11, alpha=0.01

## 1. DSGD -- momentum and scaling are set to 0
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.01  --epochs=200 --arch=vgg11 --graph=ring --momentum=0.0 --scaling=0.0 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=vgg11 --world_size=32 --skew=0.01 --graph=ring --seed=12
cd ..

## 2. GUT -- momentum is set to 0 and scaling is set to 0.9
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.01  --epochs=200 --arch=vgg11 --graph=ring --momentum=0.0 --scaling=0.9 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=vgg11 --world_size=32 --skew=0.01 --graph=ring --seed=12
cd ..

## 3. QG-DSGDm -- momentum is set to 0.9 and scaling is set to 0
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.01  --epochs=200 --arch=vgg11 --graph=ring --momentum=0.9 --scaling=0.0 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=vgg11 --world_size=32 --skew=0.01 --graph=ring --seed=12
cd ..

## 4. QG-GUTm -- momentum is set to 0.9 and scaling is set to 0.08
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.01  --epochs=200 --arch=vgg11 --graph=ring --momentum=0.9 --scaling=0.08 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=vgg11 --world_size=32 --skew=0.01 --graph=ring --seed=12
cd ..
##########################################################################################################################

# Commands to run experiments on 32-node torus, training CIFAR-10 with ResNet-20, alpha=0.1

## 1. QG-DSGDm -- momentum is set to 0.9 and scaling is set to 0
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.1  --epochs=200 --arch=resnet --graph=torus --neighbors=4 --momentum=0.9 --scaling=0.0 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=32 --skew=0.1 --graph=torus --seed=12
cd ..

## 2. QG-GUTm -- momentum is set to 0.9 and scaling is set to 0.05
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.1  --epochs=200 --arch=resnet --graph=torus --neighbors=4 --momentum=0.9 --scaling=0.05 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=32 --skew=0.1 --graph=torus --seed=12
cd ..
##########################################################################################################################

# Commands to run experiments on 32-node dyck, training CIFAR-10 with ResNet-20, alpha=0.1

## 1. QG-DSGDm -- momentum is set to 0.9 and scaling is set to 0
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.1  --epochs=200 --arch=resnet --graph=dyck --neighbors=3 --momentum=0.9 --scaling=0.0 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=32 --skew=0.1 --graph=dyck --seed=12
cd ..

## 2. QG-GUTm -- momentum is set to 0.9 and scaling is set to 0.05
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.1  --epochs=200 --arch=resnet --graph=dyck --neighbors=3 --momentum=0.9 --scaling=0.05 --devices=4 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=32 --skew=0.1 --graph=dyck --seed=12
cd ..
##########################################################################################################################

# Commands to run experiments on 16-node ring, training CIFAR-100 with ResNet-20, alpha=0.1

## 1. QG-DSGDm -- momentum is set to 0.9 and scaling is set to 0
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=200 --arch=resnet --graph=ring --momentum=0.9 --scaling=0.0 --devices=4 --dataset=cifar100 --classes=100 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..

## 2. QG-GUTm -- momentum is set to 0.9 and scaling is set to 0.005
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=200 --arch=resnet --graph=ring --momentum=0.9 --scaling=0.005 --devices=4 --dataset=cifar100 --classes=100 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=resnet --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..

##########################################################################################################################

# Commands to run experiments on 16-node ring, training Fashion-MNIST with LeNet-5, alpha=0.1

## 1. QG-DSGDm -- momentum is set to 0.9 and scaling is set to 0
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=100 --arch=lenet5 --graph=ring --momentum=0.9 --scaling=0.0 --devices=4 --dataset=fmnist --classes=10 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=lenet5 --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..

## 2. QG-GUTm -- momentum is set to 0.9 and scaling is set to 0.01
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.1  --epochs=100 --arch=lenet5 --graph=ring --momentum=0.9 --scaling=0.01 --devices=4 --dataset=fmnist --classes=10 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.1 --arch=lenet5 --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..

##########################################################################################################################

# Commands to run experiments on 16-node ring, training Imagenette with MobileNet-V2, alpha=0.1

## 1. QG-DSGDm -- momentum is set to 0.9 and scaling is set to 0
python trainer.py --lr=0.01  --batch-size=512  --world_size=16 --skew=0.1  --epochs=100 --arch=mobilenet --graph=ring --momentum=0.9 --scaling=0.0 --devices=4 --dataset=imagenette --classes=10 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.01 --arch=mobilenet --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..

## 2. QG-GUTm -- momentum is set to 0.9 and scaling is set to 0.03
python trainer.py --lr=0.01  --batch-size=512  --world_size=16 --skew=0.1  --epochs=100 --arch=mobilenet --graph=ring --momentum=0.9 --scaling=0.03 --devices=4 --dataset=imagenette --classes=10 --seed=12
cd ./outputs
python dict_to_csv.py --norm=evonorm --lr=0.01 --arch=mobilenet --world_size=16 --skew=0.1 --graph=ring --seed=12
cd ..

##########################################################################################################################