
# Convolutional Neural Networks for Sentence Classification

This is the implementation of [Convolutional Neural Networks for Sentence Classification (Y.Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181) on **Pytorch**.


## Results

Below are results corresponding to all 4 models proposed in the paper for each dataset.
Experiments have been done with a learning rate = 0.1 up to 300 epochs and all details are tuned to follow the settings defined in the paper. 

(Measure: Accuracy)

| Model        | Dataset  | MR   | TREC |
|--------------|:----------:|:------:|:----:|
| Rand         | Results  | 70.0 | 87.8 |
|              | Baseline | 76.1 | 91.2 |
| Static       | Results  | **82.4** | **93.8** |
|              | Baseline | 81.0 | 92.8 |
| Non-static   | Results  | 81.4 | **93.6** |
|              | Baseline | 81.5 | 93.6 |
| Multichannel | Results  | **81.6** | **92.6** |
|              | Baseline | 81.1 | 92.2 |


## Specification
- **model.py**: CNN sentnece classifier implementation proposed by Y. Kim.
- **run.py**: train and test a model with configs. 
 

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.2.
- GPU: GTX 1080


## Requirements

This model is based on pre-trained Word2vec([GoogleNews-vectors-negative300.bin](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)) by T.Mikolov et al.
You should download this file and place it in the root folder.

Also you should follow library requirements specified in the **requirements.txt**.

    numpy==1.12.1
    gensim==2.3.0
    scikit_learn==0.19.0


## Execution

> python run.py 

    usage: run.py [-h] [--mode MODE] [--model MODEL] [--dataset DATASET]
              [--save_model SAVE_MODEL] [--early_stopping EARLY_STOPPING]
              [--epoch EPOCH] [--learning_rate LEARNING_RATE]

    -----[CNN-classifier]-----

    optional arguments:
      -h, --help                        show this help message and exit
      --mode MODE                       train: train (with test) a model / test: test saved models
      --model MODEL                     available models: rand, static, non-static, multichannel
      --dataset DATASET                 available datasets: MR, TREC
      --save_model SAVE_MODEL           whether saving model or not (T/F)
      --early_stopping EARLY_STOPPING   whether to apply early stopping(T/F)
      --epoch EPOCH                     number of max epoch
      --learning_rate LEARNING_RATE     learning rate
