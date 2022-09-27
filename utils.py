import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import einops
from typing import Sequence, Optional


def simple_gate(x):
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    return x1 * x2


# codes from https://dacon.io/competitions/official/235951/codeshare/5906?page=1&dtype=recent
def _rotate_pointclouds(a, b, c, dots):
    mx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    my = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    mz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    m = np.dot(np.dot(mx, my), mz)
    dots = np.dot(dots, m.T)
    return dots


def dots_to_tensor_with_augmentation(
        dots: np.array,
        bins: Sequence[int] = [16, 32, 32],  # C W H
        augmentation: bool = True
) -> tf.Tensor:

    if augmentation:
        h_rot, w_rot, c_rot = tf.random.truncated_normal(
            shape=(3,), mean=0., stddev=.25
        ).numpy()
        dots = _rotate_pointclouds(
            np.radians(h_rot), np.radians(w_rot), np.radians(c_rot), dots
        )

        h_max, w_max, c_max = np.max(dots, axis=0)
        h_min, w_min, c_min = np.min(dots, axis=0)
        h_range, w_range, c_range = [
            max_ - min_ for max_, min_ in zip([h_max, w_max, c_max], [h_min, w_min, c_min])
        ]
        aug_strength = tf.random.truncated_normal(
            shape=(6,), mean=0., stddev=.25
        ).numpy()
        h_max_aug, w_max_aug, c_max_aug, h_min_aug, w_min_aug, c_min_aug = [
            val + (range_ * aug) for val, range_, aug in zip(
                [h_max, w_max, c_max, h_min, w_min, c_min],
                [h_range, w_range, c_range] + [-h_range, -w_range, -c_range],
                aug_strength
            )
        ]
        arr, _ = np.histogramdd(
            dots, bins=bins, range=(
                (h_min_aug, h_max_aug), (w_min_aug, w_max_aug), (c_min_aug, c_max_aug)
            )
        )
    else:
        arr, _ = np.histogramdd(
            dots, bins=bins
        )

    arr = np.clip(arr, a_min=0., a_max=1.)
    tensor_ = tf.convert_to_tensor(
        np.rot90(np.transpose(arr, (1, 2, 0)), k=1, axes=(0, 1)), dtype=tf.float32
    )
    return tensor_


