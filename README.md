# DHT-Net:Dynamic Hierarchical Transformer Network for Liver and Tumor Segmentation
This is the official pytorch implementation of the DHT-Net:

<img src="dhtnet.png" width="1024px" />

# Requirements
CUDA 11.0

PyTroch 1.7.0

Python 3.8

Torchvision 0.8.2

# Usage
## Installation
Install Pytorch1.7, nnUNet as below:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install nnunet
```
## Data preprocessing
1.Download [LiTS Dataset](https://competitions.codalab.org/competitions/17094) and [3DIRCADb Dataset](https://www.ircad.fr/research/3dircadb/) 

2.Preprocess the dataset according to the [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

* Dataset path

you should divide the training data,training labels,test data, test labels into four part, with the specific path (LiTS dataset as an example):

`./LiTS/data_raw/DHTNet_raw_data/TaskXXX_LiTS/imagesTr`


`./LiTS/data_raw/DHTNet_raw_data/TaskXXX_LiTS/imagesTs`


`./LiTS/data_raw/DHTNet_raw_data/TaskXXX_LiTS/labelsTr`


`./LiTS/data_raw/DHTNet_raw_data/TaskXXX_LiTS/labelsTs`


 `XXX` is the inter identifier associated with your task name.

* cd DHTNet/dhtnet/path.py

you shuould set your dataset base path `base`, your preprocessing output path `preprocessing_output_dir` and your network training output dir `network_training_output_dir_base`, as example:

`base = ./LiTS/data_raw/DHTNet_raw_data/`

`preprocessing_output_dir = ./LiTS/data_preprocessed/`

`network_training_output_dir_base = ./LiTS/Result/`

you can also specify any data path in `DHTNet/dhtnet/path.py`.

3.Opertional data preprocessing.

* cd DHTNet/dhtnet/ (set the `base_folder = base/TaskXXX_LiTS` in gen_json.py)

Run `Python gen_json.py`


* cd DHTNet/dhtnet/experiment_planning/

Run `Python nnUNet_plan_and_preprocess.py -t XXX`

 `XXX` is the inter identifier associated with your Task name.
 
## Training
* cd DHTNet/dhtnet/training/
 
`Python run_training -t XXX -f X`
 
`X` specifies which fold of the 5-fold-cross-validation is trained.
 
## Testing
  
`Python run_training -t XXX -f X -val`
  
## Inference
  
* cd DHTNet/dhtnet/inference/

`Python predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER -t xxx -f x -chk model_best`

you can ensembling the predictions from several configurations with the following command:

`Python ensenble_predictions.py -f FOLDER1 FOLDER2 ... -pp POSTPROCESSING_FILE`
   
## 5. Acknowledgements
Part of codes are reused from the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Thanks to Fabian Isensee for the codes of nnU-Net.

## Contact

Ruiyang Li(liruiyang@stu.xidian.edu.cn)
  
 
