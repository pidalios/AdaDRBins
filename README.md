<!-- 
Author: Hsuan Tsai
Email:  hsuantsai0114@gmail.com
Note:   If you want to see this repo on GitHub, please contact to Hsuan Tsai for authentication 
-->

# AdaDRBins

This repo is the official implementation of AdaDRBins.


## Usage

### Training

* Basic command:

    ```bash
    python train.py -bs 16 --epochs 50 -lr 0.000357 --lowRes [0/1] --is_bin [1/0] --workers [num_worker]
    ```

* Flags:
  
    >`-bs`: Batch size.\
    `--epochs`: Number of training epochs.\
    `-lr`:  Learning rate.\
    `--lowRes`: The value `1` means use low resolution depth as the input, `0` means use sparse depth as the input.\
    `--is_bin`: Value `0` means model will not contain bins estimator, `1` means model will contain bins estimator.\
    `--worker`: This value is your number of CPU cores.\
    For more information, please see `--help`.


* Visualization

    You can use tensordboard to visulize the results during training progress.

    ```bash
    tensorboard --logdir=./runs
    ```

* Weights

    The trained model weights will be saved to `./result/[model_name]`.

### Testing

* Basic command
    ```bash
    python evaluation.py -w /path/to/weights --mode [v/e]
    ```

* Flags

    >`--mode`: Evaluation mode.\
    `v: visulization`: Save the visualization results to `./imgs/`\
    `e: evaluation`: Evaluate model performance on NYU official testing set.

* Visualization

    The images will be saved to `./imgs/[model_name]`.

### Convert the model to ONNX format

* In `onnx_converter.py`, modify the path to your weights. Then use the command to convert the model.

    ```bash
    python onnx_converter.py
    ```

## Requirement

* Install the requirements via conda:

    ```bash
    conda env create -f ./environment.yml
    ```

* Download the preprocessed `NYU Depth V2` datasets in HDF5 formats and place them under the `data` folder(note that `data` is **outsize** of the directory). The downloading process might take an hour or so. The NYU dataset requires `32G` of storage space.
  
    ```bash
    cd data
    wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz 
    tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz 
    cd ..
    ```
