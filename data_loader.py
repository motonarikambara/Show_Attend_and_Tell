import tensorflow as tf

# Scikit-learn には役に立つさまざまなユーティリティが含まれる
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import tqdm

from caption_generator import Caption_Generator

class DataLoader():
    def init(self):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                cache_subdir=os.path.abspath('.'),
                                                origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                extract = True)
        annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

        name_of_zip = 'train2014.zip'
        if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
            image_zip = tf.keras.utils.get_file(name_of_zip,
                                                cache_subdir=os.path.abspath('.'),
                                                origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                                extract = True)
            PATH = os.path.dirname(image_zip)+'/train2014/'
        else:
            PATH = os.path.abspath('.')+'/train2014/'

        
    