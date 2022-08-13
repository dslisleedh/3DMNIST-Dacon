import tensorflow as tf
import tensorflow_addons as tfa
from models.testmodel import MyModel
from utils import *

import h5py
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def _main():
    return 0


if __name__ == '__main__':
    config = OmegaConf.load('./config.yaml')
    _main()
