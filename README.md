# Latent-Graph Convolutional Network
This repository contains the Keras/TensorFlow implementation our approach L-GCN as described in [1].

## Demo

To run the L-GCN-FC code on the RITA dataset:

```bash
python train.py --dataset rita_tts_hard
```

To run the L-GCN-LSTM code on the RITA dataset:
```bash
python train.py --dataset rita_tts_hard_lstm
```

To run the L-GCN code on the AIFB dataset:
```bash
python train.py --dataset aifb --add_node_one_hot 1 --train_split 70 --val_split 10
```

For other command line arguments please consult the help option: 
```python train.py --help```

## Models

You can choose between the following models: 
* `dense`: Basic multi-layer perceptron that supports sparse inputs
* `gcn`: Graph convolutional network (Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907), 2016)
* `efgcn`: A filter similar to the gcn filter, but here multiple adjacency matrices are used. Each edge feature has its own adjacency matrix. (be careful with highly dimensional edge features)
* `lgcn`: Latent-Graph Convolutional Network [1]


## Requirements
#### Minimal
If you only want to run the code, and perform no preprocessing
* Python 2.7.12
* TensorFlow (=>1.2.0)
* Keras (=>2.0.5)
* scikit-learn (=>0.18.1)

#### Additional
* Pandas (=>0.19.2)
* Seaborn (=>0.7.1)

## References
[1] W.B.W Vos, P. Bloem, F.M Jansen, A.L. De Souza, Z. Sun, "End-to-end learning of latent edge weights for Graph Convolutional Networks", (2017)
