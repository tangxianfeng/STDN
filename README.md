# STDN (Spatial-Temporal Dynamic Network)

## About
Source code of the paper [Revisiting Spatial-Temporal Similarity: A Deep Learning Framework for Traffic Prediction](https://arxiv.org/abs/1803.01254)

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{yao2019revisiting,
  title={Revisiting Spatial-Temporal Similarity: A Deep Learning Framework for Traffic Prediction},
  author={Yao, Huaxiu and Tang, Xianfeng and Wei, Hua and Zheng, Guanjie and Li, Zhenhui},
  booktitle={2019 AAAI Conference on Artificial Intelligence (AAAI'19)},
  year={2019} 
}
```

## Installation
Requirements

  - Python 3.6 (Recommend Anaconda)
  - Ubuntu 16.04.3 LTS
  - Keras >= 2.0.8
  - tensorflow-gpu (or tensorflow) == 1.3.0 ([install guide](https://www.tensorflow.org/versions/r1.0/install/install_linux))

## Usage
  - Download all codes (*\*.py*) and put them in the same folder (let's name it "stdn") (*stdn/\*.py*)
  - Create "data" folder in the same folder (*stdn/data/*)
  - Create "hdf5s" folder for logs (if not exist) (*stdn/hdf5s/*)
  - Download and extract all data files (*\*.npz*) from data.zip and put them in "data" folder (*stdn/data/\*.npz*)
  - Open terminal in the same folder (*stdn/*)
  - Run with "python main.py" for NYC taxi dataset, or "python main.py --dataset=bike" for NYC bike dataset
  ```
  python main.py
  ```
  ```
  python main.py --dataset=bike
  ```
  - Check the output results (RMSE and MAPE). Models are saved to "hdf5s" folder for further use.
  
## Hyperparameters:
Please check the hyperparameters defined in main.py