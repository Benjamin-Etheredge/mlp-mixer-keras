# MLP-Mixer in Keras
<img align='center' src='mlp-mixer.png'>


This is a simple keras implementation of MLP-Mixer. MLP-Mixer is an almost exclusivly multi-layer perceptions approach to vision like tasks.
## Install
```bash
$ pip install mlp-mixer-keras
```

## Example usage
```python
from mlp_mixer_keras import MlpMixerModel 
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
model = MlpMixerModel(input_shape=x_train.shape[1:],
                      num_classes=len(np.unique(y_train)), 
                      num_blocks=4, 
                      patch_size=8,
                      hidden_dim=32, 
                      tokens_mlp_dim=64,
                      channels_mlp_dim=128,
                      use_softmax=True)
model.compile(loss='sparse_categorical_crossentropy', metrics='acc')
model.fit(x_train, y_train, validation_data=(x_test, y_test))
```


## References

### MLP-Mixer: An all-MLP Architecture for Vision
Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy, [*MLP-Mixer: An all-MLP Architecture for Vision*](https://arxiv.org/abs/2105.01601)

```bibtex
@misc{tolstikhin2021mlpmixer,
      title={MLP-Mixer: An all-MLP Architecture for Vision}, 
      author={Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
      year={2021},
      eprint={2105.01601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### MLP-Mixer: An all-MLP Architecture for Vision (Machine Learning Research Paper Explained)
Excellent Yannic Kilcher explainer [video](https://youtu.be/7K4Z8RqjWIk). 

### MLP Mixer - Pytorch
A pytorch implementation of MLP-Mixer. This repo helped a alot as I learned the ways of making a nice github repo for a project.

Phil Wang - [lucidrains](https://github.com/lucidrains)

 [*MLP Mixer - Pytorch*](https://github.com/lucidrains/mlp-mixer-pytorch)