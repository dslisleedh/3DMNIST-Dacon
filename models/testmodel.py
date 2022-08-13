from models.layers import Block
from utils import simple_gate
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Sequence, Optional


class MyModel(tf.keras.models.Model):
    def __init__(
            self, n_filters: Sequence[int], n_layers: Sequence[int], strides: Sequence[int],
            n_labels: int, drop_rate: Sequence[float], act=simple_gate, n_heads: int = 8,
            **kwargs
    ):
        super(MyModel, self).__init__(**kwargs)
        assert len(n_filters) == len(strides) == len(n_layers)
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.strides = strides
        self.n_labels = n_labels
        self.drop_rate = drop_rate
        self.act = act
        self.n_heads = n_heads

        self.feature_extractor = tf.keras.Sequential([
            Block(
                n_filter, n_layer, stride, self.drop_rate, self.act, self.n_heads
            ) for n_filter, n_layer, stride in zip(self.n_filters, self.n_layers, self.strides)
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(
                self.n_labels, activation='softmax', kernel_initializer=tf.keras.initializers.zeros()
            )
        ])

    def call(self, inputs, training=None, mask=None):
        return self.classifier(self.feature_extractor(inputs))
