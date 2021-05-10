# MLP-Mixer in Keras
<img align='center' src='mlp-mixer.png>
This is a keras implementation of MLP-Mixer

## Example usage
```
from mlp_mixer_keras import MlpMixerModel 
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
model = MlpMixerModel(input_shape = x_train.shape[1:],
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