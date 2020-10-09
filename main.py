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
    def __init__(self):
        self.captiongenerator = Caption_Generator() 

    def train(self):
        self.caption_genarator.model()
        start_epoch = 0
        ckpt_manager = self.caption_genarator.ckpt_manager
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

        EPOCHS = 20
        for epoch in range(start_epoch, EPOCHS):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(self.dataset):
                batch_loss, t_loss = self.caption_genarator.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

            if epoch % 5 == 0:
                ckpt_manager.save()

            print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                                total_loss/num_steps))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    def test(self):
        # 検証用セットのキャプション
        rid = np.random.randint(0, len(img_name_val))
        image = img_name_val[rid]
        real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
        result = self.caption_genarator.evaluate(image)

        print ('Real Caption:', real_caption)
        print ('Prediction Caption:', ' '.join(result))

        image_url = 'https://tensorflow.org/images/surf.jpg'
        image_extension = image_url[-4:]
        image_path = tf.keras.utils.get_file('image'+image_extension,
                                            origin=image_url)

        result = self.evaluate(image_path)
        print ('Prediction Caption:', ' '.join(result))
        # 画像を開く
        Image.open(image_path)
