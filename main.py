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

from caption_genarator import Caption_Generator

class ShowAttendandTell():
    def init():
        self.captiongenerator = Caption_Generator() 