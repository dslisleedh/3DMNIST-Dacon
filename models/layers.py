import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import einops
from utils import simple_gate

from typing import Sequence


class DropTime(tf.keras.layers.Layer):
    def __init__(self, drop_rate: float):
        super(DropTime, self).__init__()
        self.drop_rate = drop_rate

    def call(self, inputs, training):
        def call_train():
            b, t, c = tf.shape(inputs)
            epsilon = K.random_bernoulli(
                shape=(b, t, c), p=1. - self.drop_rate, dtype=tf.float32
            )
            return inputs * epsilon

        def call_test():
            return inputs

        return K.in_train_phase(
            call_train, call_test, training=training
        )


class DropPath(tf.keras.layers.Layer):
    def __init__(self, layer, drop_rate: float):
        super(DropPath, self).__init__()
        self.layer = layer
        self.survival_prob = 1. - drop_rate

    def call(self, inputs, training):
        if training:
            if tf.equal(
                K.random_bernoulli(shape=(), p=self.survival_prob),
                1.
            ):
                return self.layer(inputs) / self.survival_prob
            else:
                return inputs
        else:
            return self.layer(inputs)


class MultiLayerPerceptron(tf.keras.layers.Layer):
    def __init__(
            self, act: simple_gate, expansion_rate: int = 4, **kwargs
    ):
        super(MultiLayerPerceptron, self).__init__(**kwargs)
        self.act = act
        self.expansion_rate=expansion_rate

    def build(self, input_shape):
        super(MultiLayerPerceptron, self).build(input_shape)
        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(
                input_shape[-1] * self.expansion_rate, activation=self.act,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.Dense(
                input_shape[-1],
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(
            self, n_heads: int = 8, **kwargs
    ):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.n_heads = n_heads

    def build(self, input_shape):
        super(MultiHeadSelfAttention, self).build(input_shape)
        self.to_qkv = tf.keras.layers.Dense(
            input_shape[-1] * 3, activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.to_out = tf.keras.layers.Dense(
            input_shape[-1], activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.scale = tf.Variable(
            tf.sqrt(input_shape[-1] / self.n_heads), trainable=False
        )

    def to_heads(self, x):
        return einops.rearrange(
            x, 'b t (h c) -> b h t c',
            h=self.n_heads
        )

    def call(self, inputs, *args, **kwargs):
        q, k, v = tf.split(
            self.to_qkv(inputs),
            num_or_size_splits=3,
            axis=-1
        )
        q, k, v = [self.to_heads(qkv) for qkv in [q, k, v]]
        attention = tf.nn.softmax(tf.matmul(q / self.scale, k, transpose_b=True))

        out = self.to_out(
            einops.rearrange(
                tf.matmul(attention, v), 'b h t c -> b t (h c)'
            )
        )
        return out


class Block(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: int, n_layers: int, strides: int,
            drop_rate: Sequence[float], act=simple_gate, n_heads: int = 8, **kwargs
    ):
        super(Block, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.drop_rate = drop_rate

        self.conv = tf.keras.layers.Conv2D(
            n_filters, (3, 3), activation=None,
            strides=(strides, strides), padding='SAME',
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.dwc = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=(1, 1), padding='SAME',
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.droptime = DropTime(drop_rate[0])
        self.attentions = [
            DropPath(MultiHeadSelfAttention(n_heads), self.drop_rate[1]) for _ in range(n_layers)
        ]
        self.mlps = [
            DropPath(MultiLayerPerceptron(act=act), self.drop_rate[1]) for _ in range(n_layers)
        ]

    def call(self, inputs, training):
        features = self.dwc(self.conv(inputs))
        _, h, w, _ = tf.shape(features)
        features = einops.rearrange(
            features, 'b h w c -> b c (h w)'
        )
        features = self.droptime(features)
        for attention, mlp in zip(self.attentions, self.mlps):
            drop_state = tf.random.uniform(shape=(), minval=0., maxval=1.)
            if tf.less(drop_state, self.drop_rate[1]):
                continue
            res = features + attention(features)
            res += mlp(res)

        features = tf.reshape(
            tf.transpose(features, (0, 2, 1)), (-1, h, w, self.n_filters)
        )
        return features
