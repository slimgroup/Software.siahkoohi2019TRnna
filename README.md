# Neural network augmented wave-equation simulation

Codes for generating results in Siahkoohi, A., Louboutin, M. and Herrmann, F.J., 2019. Neural network augmented wave-equation simulation. arXiv preprint [arXiv:1910.00925](https://arxiv.org/abs/1910.00925).



## Prerequisites

This code has been tested on Deep Learning AMI (Amazon Linux) Version 11.0 on Amazon Web Services (AWS). We used `g3.4xlarge` instance. Also, we use GCC compiler version 7.3.0.

This software is based on [Devito-3.2.0](https://github.com/opesci/devito/releases/tag/v3.2.0), [ODL-0.7.0](https://github.com/odlgroup/odl/releases/tag/v0.7.0), and [TensorFlow-1.10.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.10.0). Follow the steps below to install the necessary libraries:

```bash
cd $HOME
git clone https://github.com/alisiahkoohi/NN-augmented-wave-sim
git clone --branch v0.7.0 https://github.com/odlgroup/odl.git
git clone --branch v3.2.0 https://github.com/devitocodes/devito.git

cd $HOME/devito
conda env create -f environment.yml
source activate devito
pip install  -e .
export DEVITO_ARCH=gnu
export OMP_NUM_THREADS=16
export DEVITO_OPENMP=1

cd $HOME/odl
pip install --user -e .

pip install tensorflow-gpu==1.10.0
pip install h5py
```

## Dataset

The Marmousi model we use is obtained from [Devito Codes project](https://github.com/devitocodes) and will be automatically downloaded and placed at `./vel-model/` directory.

## Script descriptions

`RunTraining.sh`\: script for running training on `AWS`. It will make `model/` and `data/` directory in `$HOME` for storing training/testing data and saved neural net checkpoints and final results, respectively. Next, it will train a neural net for the experiment.

`RunTraining_shared_weights.sh`\: script for running training on `AWS` for the case where networks share their weights. It will make `model/` and `data/` directory in `$HOME` for storing training/testing data and saved neural net checkpoints and final results, respectively. Next, it will train a neural net for the experiment.

`RunTesting.sh`\: script for testing the trained neural net on `AWS`. 

`RunTesting.sh`\: script for testing the trained neural net on `AWS` for the case where networks share their weights. 

`src/main.py`\: constructs `LearnedWaveSim` class using given arguments in `RunTraining.sh`\, defined in `model.py` and calls `train` function in the defined  `LearnedWaveSim` class.

`src/model.py`: includes `LearnedWaveSim` class definition, which involves `train` and `test` functions.

`src/main_shared_weights.py`\: constructs `LearnedWaveSim` class using given arguments in `RunTraining.sh`\, defined in `model.py` and calls `train` function in the defined  `LearnedWaveSim` class, for the case where networks share their weights.

`src/model_shared_weights.py`: includes `LearnedWaveSim` class definition, which involves `train` and `test` functions, for the case where networks share their weights.

`show_prediction.py`\: Plotting the results.

### Running the code

To perform training on AWS, run:

```bash
# Running in GPU

bash RunTraining.sh

```

To evaluated the pre-trained and transfer-trained neural net on test dataset run the following. It will automatically load the latest checkpoint saved for both neural nets.

```bash
# Running in GPU

bash RunTesting.sh

```

To generate and save figures shown in paper for gradient conditioning, run:

```bash

bash src/genFigures.sh

```

The saving directory can be changed by modifying `savePath` variable in `src/genFigures.sh`\.


## Questions

Please contact alisk@gatech.edu for further questions.

## Acknowledgments

The authors thank Xiaowei Hu for his open-access [repository](https://github.com/xhujoy/CycleGAN-tensorflow) on GitHub. Our software implementation built on this work.

## Author

Ali Siahkoohi
