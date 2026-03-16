# SAUID

This is the Pytorch implementation of IVC paper 'Semantic-assisted Unpaired Image Dehazing'. 

The weights are available in https://drive.google.com/drive/folders/1CXaDFA_tw02onLdKmORGcJdIp2JpKDX4?usp=sharing 

### 1. Training 
1) Prepare the training datasets following the operations in the Datasets part. 
2) Add a config file 'config.yml' in the checkpoints folder. We have provided example checkpoints folder and config files in `./checkpoints/train_example`. Make sure TRAIN_CLEAN_FLIST and TRAIN_HAZY_FLIST are right. 
3) Train the model, for example:
```
python train.py --model 1 --checkpoints ./checkpoints/train_example
```

### 2. Testing
1)Prepare the testing datasets following the operations in the Datasets part.
2)Put the trained weight in the checkpoint folder 
3)Add a config file 'config.yml' in your checkpoints folder. We have provided example checkpoints folder and config files in `./checkpoints/`, 
4)Test the model, for example:
```
python test.py --model 1 --checkpoints ./checkpoints/test_revide
```
For quick testing, you can download the checkpoint on real-world frames and put it to the corresponding folder `./checkpoints/test_real` and run test on our example frames directly using

```
python test.py --model 1 --checkpoints ./checkpoints/test_real
```
