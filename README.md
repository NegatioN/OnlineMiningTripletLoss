# Triplet Loss and Online Triplet Mining in PyTorch (GPU)
PyTorch conversion of the excellent post on the [same topic in Tensorflow](https://omoindrot.github.io/triplet-loss).
Simply an implementation of a triple loss with online mining of candidate triplets used in semi-supervised learning.

### Usage
Include the `triplet_loss.py` file in your project and include either `batch_hard_triplet_loss` or `batch_all_triplet_loss`.

Example usage:
```
from triplet_loss import batch_hard_triplet_loss

labels = torch.randint(5) # our five labels

embeddings = model(labels)

loss = batch_hard_triplet_loss(labels, embeddings, margin=0.2)
loss.backward()
# and so on
```

### Tests
```
pip install -r requirements.txt
python3 -m pytest test_triplet_loss.py
```

## References
* [Triplet Loss and Online Triplet Mining in Tensorflow](https://github.com/omoindrot/tensorflow-triplet-loss)
* [Facenet paper](https://arxiv.org/abs/1503.03832)
* [adambielski's nice implementation](https://github.com/adambielski/siamese-triplet) (unfortunately context switches between CPU / GPU)
