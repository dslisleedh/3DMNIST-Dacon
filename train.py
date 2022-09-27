import tensorflow as tf
import tensorflow_addons as tfa
from models.testmodel import MyModel
from utils import *

import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
from omegaconf import OmegaConf
import itertools
from functools import partial
import time
import os


def _main():
    train_time = time.localtime(time.time())
    work_path = f'./logs/{train_time[0]}{train_time[1]:02}{train_time[2]:02}_{train_time[3]:02}{train_time[4]:02}{train_time[5]:02}'
    os.makedirs(work_path)
    config.work_path = work_path
    OmegaConf.save(
        config, work_path + '/hparams.yaml'
    )

    train_all = h5py.File(config.load_path+'train.h5', 'r')
    train_df = pd.read_csv(config.load_path+'train.csv')
    test_all = h5py.File(config.load_path+'train.h5', 'r')
    sub_df = pd.read_csv(config.load_path+'sample_submission.csv')

    train_list = [np.array(train_all[str(i)]) for i in tqdm(train_df["ID"])[:40000]]
    valid_list = [np.array(train_all[str(i)]) for i in tqdm(train_df["ID"])[40000:]]
    test_list = [np.array(test_all[str(i)]) for i in tqdm(sub_df["ID"])]
    train_label = train_df['label'][:40000]
    valid_label = train_df['label'][40000:]

    no_aug_prep = partial(dots_to_tensor_with_augmentation, augmentation=False)
    train_gen = tf.data.Dataset.from_generator(
        lambda: itertools.zip_longest(train_list, train_label),
        output_types=(tf.float32, tf.float32), output_shape=([None, 3], [])
    ).map(  # map after batch is faster, but implemented this way for convenience
        lambda x, y: (tf.py_function(dots_to_tensor_with_augmentation, inp=[x], Tout=tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(1000).batch(config.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    valid_gen = tf.data.Dataset.from_generator(
        lambda: itertools.zip_longest(valid_list, valid_label),
        output_types=(tf.float32, tf.float32), output_shape=([None, 3], [])
    ).map(
        lambda x, y: (tf.py_function(no_aug_prep, inp=[x], Tout=tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    test_gen = tf.data.Dataset(
        lambda: test_list, output_types=tf.float32, output_shapes=[None, 3]
    ).map(
        lambda x: tf.py_function(no_aug_prep, inp=[x], Tout=tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    model = MyModel(
        config.n_filters, config.n_layers, config.strides, config.n_labels,
        config.drop_rate, simple_gate if config.act == 'simple_gate' else tf.nn.gelu, config.n_heads
    )
    lr_scheduler = tf.keras.experimental.CosineDecay(
        config.learning_rate, 40000 * config.epochs,
        alpha=1e-5
    )
    optimizer = tf.keras.optizmiers.Adam(
        learning_rate=lr_scheduler, clipvalue=1.
    )
    model.compile(
        optimizer, loss='sparse_categorical_crossentropy', metrics=['acc']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config.workpath, update_freq='batch'
        )
    ]
    history = model.fit(
        train_gen, validation_data=valid_gen, epochs=config.epochs, callbacks=callbacks
    )
    model.save_weights(config.work_path + '/model_weights')

    preds = model.predict(test_gen)

    sub_file = sub_df['ID']
    sub_file['label'] = tf.argmax(preds, axis=-1).numpy()

    best_score = max(history.history['acc'])
    return best_score


if __name__ == '__main__':
    config = OmegaConf.load('./config.yaml')
    np.random.set_state(config.seed)
    tf.random.set_seed(config.seed)
    best_score = _main()

    print('Train end !!!')
    print(f'Best validation score: {best_score}')
