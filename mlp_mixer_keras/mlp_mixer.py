from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    Dense,
    Conv2D,
    GlobalAveragePooling1D,
    Layer,
    LayerNormalization,
    Permute,
    Softmax,
    Activation,
)


class MlpBlock(Layer):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation=None,
        **kwargs
    ):
        super(MlpBlock, self).__init__(**kwargs)

        if activation is None:
            activation = keras.activations.gelu

        self.dim = dim
        self.hidden_dim = dim
        self.dense1 = Dense(hidden_dim)
        self.activation = Activation(activation)
        self.dense2 = Dense(dim)

    def call(self, inputs):
        x = inputs
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x

    def compute_output_shape(self, input_signature):
        return (input_signature[0], self.dim)

    def get_config(self):
        config = super(MlpBlock, self).get_config()
        return {
            **config,
            'dim': self.dim,
            'hidden_dim': self.hidden_dim
        }


class MixerBlock(Layer):
    def __init__(
        self,
        num_patches: int,
        channel_dim: int,
        token_mixer_hidden_dim: int,
        channel_mixer_hidden_dim: int = None,
        activation=None,
        **kwargs
    ):
        super(MixerBlock, self).__init__(**kwargs)

        if activation is None:
            activation = keras.activations.gelu

        if channel_mixer_hidden_dim is None:
            channel_mixer_hidden_dim = token_mixer_hidden_dim

        self.norm1 = LayerNormalization(axis=1)
        self.permute1 = Permute((2, 1))
        self.token_mixer = MlpBlock(num_patches, token_mixer_hidden_dim, name='token_mixer')

        self.permute2 = Permute((2, 1))
        self.norm2 = LayerNormalization(axis=1)
        self.channel_mixer = MlpBlock(channel_dim, channel_mixer_hidden_dim, name='channel_mixer')

        self.skip_connection1 = Add()
        self.skip_connection2 = Add()

    def call(self, inputs):
        x = inputs
        skip_x = x
        x = self.norm1(x)
        x = self.permute1(x)
        x = self.token_mixer(x)

        x = self.permute2(x)

        x = self.skip_connection1([x, skip_x])
        skip_x = x

        x = self.norm2(x)
        x = self.channel_mixer(x)

        x = self.skip_connection2([x, skip_x])  # TODO need 2?

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def MlpMixerModel(
        input_shape: int,
        num_classes: int,
        num_blocks: int,
        patch_size: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int = None,
        use_softmax: bool = False,
):
    height, width, _ = input_shape

    if channels_mlp_dim is None:
        channels_mlp_dim = tokens_mlp_dim

    num_patches = (height*width)//(patch_size**2)  # TODO verify how this behaves with same padding

    inputs = keras.Input(input_shape)
    x = inputs

    x = Conv2D(hidden_dim,
               kernel_size=patch_size,
               strides=patch_size,
               padding='same',
               name='projector')(x)

    x = keras.layers.Reshape([-1, hidden_dim])(x)

    for _ in range(num_blocks):
        x = MixerBlock(num_patches=num_patches,
                       channel_dim=hidden_dim,
                       token_mixer_hidden_dim=tokens_mlp_dim,
                       channel_mixer_hidden_dim=channels_mlp_dim)(x)

    x = GlobalAveragePooling1D()(x)  # TODO verify this global average pool is correct choice here

    x = LayerNormalization(name='pre_head_layer_norm')(x)
    x = Dense(num_classes, name='head')(x)

    if use_softmax:
        x = Softmax()(x)
    return keras.Model(inputs, x)
