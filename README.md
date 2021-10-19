# online_triplet_loss
> PyTorch conversion of the excellent post on the <a href='https://omoindrot.github.io/triplet-loss'>same topic in Tensorflow</a>. Simply an implementation of a triple loss with online mining of candidate triplets used in semi-supervised learning.


## Install

`pip install online_triplet_loss`

Then import with:
`from online_triplet_loss.losses import *`

PS: Requires Pytorch version 1.1.0 or above to use.

## How to use

In these examples I use a really large margin, since the embedding space is so small. A more realistic margins seems to be between `0.1 and 2.0`

```
from torch import nn
import torch

model = nn.Embedding(10, 10)
```

```
#from online_triplet_loss.losses import *
labels = torch.randint(high=10, size=(5,)) # our five labels

embeddings = model(labels)
print('Labels:', labels)
print('Embeddings:', embeddings)
loss = batch_hard_triplet_loss(labels, embeddings, margin=100)
print('Loss:', loss)
loss.backward()
```

    Labels: tensor([6, 1, 3, 6, 6])
    Embeddings: tensor([[-1.1335,  0.3364, -3.0174, -0.8732, -0.9301,  1.3619,  0.3746,  0.0457,
              0.0180, -0.4500],
            [ 1.0757, -0.8420, -0.7630, -0.0746,  1.1545,  0.4017,  0.5587,  1.7947,
              0.1992, -2.2288],
            [ 0.2646,  1.2383,  0.1949,  0.5743, -0.8460, -0.9929, -2.0350,  0.2095,
              0.2129, -0.4855],
            [-1.1335,  0.3364, -3.0174, -0.8732, -0.9301,  1.3619,  0.3746,  0.0457,
              0.0180, -0.4500],
            [-1.1335,  0.3364, -3.0174, -0.8732, -0.9301,  1.3619,  0.3746,  0.0457,
              0.0180, -0.4500]], grad_fn=<EmbeddingBackward>)
    Loss: tensor(95.1271, grad_fn=<MeanBackward0>)


```
#from online_triplet_loss.losses import *
embeddings = model(labels)
print('Labels:', labels)
print('Embeddings:', embeddings)
loss, fraction_pos = batch_all_triplet_loss(labels, embeddings, squared=False, margin=100)
print('Loss:', loss)
loss.backward()
```

    Labels: tensor([6, 1, 3, 6, 6])
    Embeddings: tensor([[-1.1335,  0.3364, -3.0174, -0.8732, -0.9301,  1.3619,  0.3746,  0.0457,
              0.0180, -0.4500],
            [ 1.0757, -0.8420, -0.7630, -0.0746,  1.1545,  0.4017,  0.5587,  1.7947,
              0.1992, -2.2288],
            [ 0.2646,  1.2383,  0.1949,  0.5743, -0.8460, -0.9929, -2.0350,  0.2095,
              0.2129, -0.4855],
            [-1.1335,  0.3364, -3.0174, -0.8732, -0.9301,  1.3619,  0.3746,  0.0457,
              0.0180, -0.4500],
            [-1.1335,  0.3364, -3.0174, -0.8732, -0.9301,  1.3619,  0.3746,  0.0457,
              0.0180, -0.4500]], grad_fn=<EmbeddingBackward>)
    tensor(94.9947, grad_fn=<DivBackward0>) tensor(1.)
    Loss: tensor(94.9947, grad_fn=<DivBackward0>)


## References
* [Triplet Loss and Online Triplet Mining in Tensorflow](https://github.com/omoindrot/tensorflow-triplet-loss)
* [Facenet paper](https://arxiv.org/abs/1503.03832)
* [adambielski's nice implementation](https://github.com/adambielski/siamese-triplet) (unfortunately context switches between CPU / GPU)
