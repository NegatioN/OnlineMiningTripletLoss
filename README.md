# Triplet Loss and Online Triplet Mining in PyTorch (GPU)
PyTorch conversion of the excellent post on the [same topic in Tensorflow](https://omoindrot.github.io/triplet-loss).
Simply an implementation of a triple loss with online mining of candidate triplets used in semi-supervised learning.


### Tests
```
pip install -r requirements.txt
python3 -m pytest test_triplet_loss.py
```

## References
* [Triplet Loss and Online Triplet Mining in Tensorflow](https://github.com/omoindrot/tensorflow-triplet-loss)
* [Facenet paper](https://arxiv.org/abs/1503.03832)
* [adambielski's nice implementation](https://github.com/adambielski/siamese-triplet) (unfortunately context switches between CPU / GPU)
