# Data preprocessing

Follow the script to create a .h5 file from the ACDC zip files (from training and testing data).
The attributes_ACDC.csv files contain the attributes information (LVEDV, RVEDV, MyoEDV).

# Training process

## Description

The training process is in two steps
1. Train the AR-SIVAE 
2. Train the MLP to do either binary or multi-class classfifcation 

Before starting training, please follow the pre-processing steps and adpat the following parameters in the config files to load the data

```bash
data_loader:
  module_name: data.cardiac_attributes_loader
  class_name: CardiacLoader
  params:
    args:
      patch_path: #path to the .h5 file
      attributes_path: # path to the folder with the files attributes_ACDC.csv and testing_attributes_ACDC.csv
```


### Train AR-SIVAE

   ```bash
   python core/Main.py --config_path projects/AR-SIVAE/config/"method names"/config.yaml
   ```
The folder projects/AR-SIVAE/config contains the config used for obtaining the weights in the paper. 

### Train MLP

After training the different methods, you have to copy the weights and the config file of the corresponding model into folder and modify the config file accordingly.
   ```bash:wq
   python core/Main.py --config_path projects/MLP/config/config_bin.yaml
   ```
In config_bin.yaml (or config_multi), you can configure which pretrained VAE you  are using in the config file.

### Test

To test with pretrained weight any method, you can replace in the config file the following:

```bash
experiment:
  name: MLP
  task: test
  weights: 'path_to_my_weight.pt'
```
