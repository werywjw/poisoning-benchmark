# Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks

**Updated to include new benchmarks on TinyImageNet dataset (November 2020)**

This repository is the official implementation of [Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks](https://arxiv.org/abs/2006.12557). 

### CIFAR-10
##### Transfer Learning

| Attack                        | White-box (%)      | Black-box (%)|
| ------------------            |-------------------:|-------------:|
|Feature Collision              | 22.0               | 7.0          |
|Convex Polytope                | 33.0               | 7.0          |
|Bullseye Polytope              | 85.0               | 8.5          |
|Clean Label Backdoor           | 5.0                | 6.5          |
|Hidden Trigger Backdoor        | 10.0               | 9.5          |

    
##### From Scratch Training

| Attack                    | ResNet-18 (%)     | MobileNetV2 (%)   | VGG11 (%) | Average (%)|
| --------------------------| -----------------:|------------------:|----------:|-----------:|
|Feature Collision          |  0                |  1                |  3        |  1.33      |   
|Convex Polytope            |  0                |  1                |  1        |  0.67      |   
|Bullseye Polytope          |  3                |  3                |  1        |  2.33      |   
|Witches' Brew              |  45               |  25               |  8        |  26.00     |   
|Clean Label Backdoor       |  0                |  1                |  2        |  1.00      | 
|Hidden Trigger Backdoor    |  0                |  4                |  1        |  2.67      | 

***

### TinyImageNet
##### Transfer Learning

| Attack                        | White-box (%)      | Black-box (%)|
| ------------------            |-------------------:|-------------:|
|Feature Collision              | 49.0               | 32.0         |
|Convex Polytope                | 14.0               | 1.0          |
|Bullseye Polytope              | 100.0              | 10.5         |
|Clean Label Backdoor           | 3.0                | 1.0          |
|Hidden Trigger Backdoor        | 3.0                | 0.5          |
    
##### From Scratch Training

| Attack                    | VGG11 (%) |
| --------------------------|----------:|
|Feature Collision          |  4        |  
|Convex Polytope            |  0        |  
|Bullseye Polytope          |  44       |  
|Witches' Brew              |  32       |  
|Clean Label Backdoor       |  0        |
|Hidden Trigger Backdoor    |  0        |

###### For more information on each attack consult [our paper](https://arxiv.org/abs/2006.12557) and the original sources listed there.

---

# Getting Started:
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Then download the [TinyImageNet Dataset](https://tiny-imagenet.herokuapp.com/). (Additionally available on our [drive](https://drive.google.com/drive/folders/1MMebJznKStXcFT31MKyyec2GMWcsrwtP?usp=sharing)). In [learning_module.py](learning_module.py), change the line
```
TINYIMAGENET_ROOT = "/fs/cml-datasets/tiny_imagenet"
```
accordingly, to point to the unzipped TinyImageNet directory. (It is left in this repo to match our filesystem, and will likely not work with yours.)

## Pre-trained Models

Pre-trained checkpoints used in this benchmark can be downloaded from [here](https://drive.google.com/drive/folders/1MMebJznKStXcFT31MKyyec2GMWcsrwtP?usp=sharing). They should be copied into the [pretrained_models](pretrained_models) folder (which is empty until downloaded models are added).

---
## Testing

To test a model, run:

```
python3 test_model.py --model 'resnet18'  --model_path 'pretrained_models/ResNet18_CIFAR10_adv.pth' 
```    

### 💻 Record update (02.March.2024)
```
wery@Werys-MBP poisoning-benchmark % python3 test_model.py --model 'resnet18'  --model_path 'pretrained_models/ResNet18_CIFAR10_adv.pth'    
20240302 01:31:07 test_model.py main() running.
Files already downloaded and verified
Files already downloaded and verified
20240302 01:40:43  Training accuracy:  24.908
20240302 01:40:43  Natural accuracy:  25.12
```
See the code for additional optional arguments.

## Crafting Poisons With Our Setups
See [How To](how_to.md) for full details and sample code.

## Evaluating A Single Batch of Poison Examples
We have left one sample folder of poisons in poison_examples.

### 💻 Record update (07.March.2024)
```eval
python3 poison_test.py --model 'resnet18' --model_path 'pretrained_models/ResNet18_CIFAR10_adv.pth' --poisons_path 'poison_setups/'
```
This allows users to test their poisons in a variety of settings, not only the benchmark setups. See the file [poison_test.py](poison_test.py) for a comprehensive list of arguments.

## Benchmarking A Backdoor or Triggerless Attack
To compute benchmark scores, craft 100 batches of poisons using the setup pickles (for transfer learning: poison_setups_transfer_learning.pickle, for from-scratch training: poison_setups_from_scratch.pickle), and run the following. 

*Important Note:* In order to be on the leaderboard, new submissions must host their poisoned datasets online for public access, so results can be corroborated without producing new poisons. Consider a Dropbox or GoogleDrive folder with all 100 batches of poisons.

For one trial of transfer learning poisons:

### 💻 Record update (07.March.2024)
```eval
python3 benchmark_test.py --poisons_path 'poison_setups/'  --dataset 'cifar10'
```

```
wery@Werys-MBP poisoning-benchmark % python3 benchmark_test.py --poisons_path 'poison_setups/'  --dataset 'cifar10' 
20240307 18:19:39 benchmark_test.py running.
Testing poisons from poison_setups/, in the transfer learning setting...

Transfer learning test:
Namespace(from_scratch=False, poisons_path='poison_setups/', dataset='cifar10', output='output_default', pretrain_dataset='cifar100', ffe=True, num_poisons=25, trainset_size=2500, lr=0.01, lr_schedule=[30], epochs=40, image_size=32, patch_size=5, train_augment=True, normalize=True, weight_decay=0.0002, batch_size=128, lr_factor=0.1, val_period=20, optimizer='SGD')
20240307 18:19:39 poison_test.py main() running.
25  poisons in this trial.
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
==> Training network...
20240307 18:29:19  Epoch:  19 , Loss:  0.7128323137760162 , Training acc:  74.92 , Natural accuracy:  68.97 , poison success:  False
Adjusting learning rate  0.01 -> 0.001
20240307 18:38:51  Epoch:  39 , Loss:  0.6499185979366302 , Training acc:  76.92 , Natural accuracy:  69.89 , poison success:  False
20240307 18:40:26  Training ended at epoch  39 , Natural accuracy:  69.89
20240307 18:40:26  poison success:  False  poisoned_label:  7  prediction:  4
20240307 18:40:26 poison_test.py main() running.
25  poisons in this trial.
```

For one trial of from-scratch training poisons:
### 💻 Record update (07.March.2024)
```eval
python3 benchmark_test.py --poisons_path 'poison_setups/' --dataset 'cifar10' --from_scratch
```

To benchmark 100 batches of poisons, run
```eval
bash benchmark_all.sh <path_to_directory_with_100_batches> 
``` 
or
```eval
bash benchmark_all.sh <path_to_directory_with_100_batches> from_scratch
``` 
