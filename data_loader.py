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
    def __init__(self):
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

        # json ファイルの読み込み
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # ベクトルにキャプションと画像の名前を格納
        all_captions = []
        all_img_name_vector = [] 

        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            all_img_name_vector.append(full_coco_image_path)
            all_captions.append(caption)

        # captions と image_names を一緒にシャッフル
        # random_state を設定
        self.train_captions, self.img_name_vector = shuffle(all_captions,
                                                all_img_name_vector,
                                                random_state=1)


    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    # numpy ファイルをロード
    def map_func(self, img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        return img_tensor, cap
    
    def dataprepare(self):

        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output

        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        # 重複のない画像を取得
        encode_train = sorted(set(self.img_name_vector))

        # batch_size はシステム構成に合わせて自由に変更可能
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

        for img, path in tqdm(image_dataset):
        batch_features = self.image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())
    
    # データセット中の一番長いキャプションの長さを検出
    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    def tokenize(self):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>",
                                                        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(self.train_captions)
        train_seqs = self.tokenizer.texts_to_sequences(self.train_captions)

        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        # トークン化したベクトルを生成
        train_seqs = tokenizer.texts_to_sequences(self.train_captions)

        # キャプションの最大長に各ベクトルをパディング
        # max_length を指定しない場合、pad_sequences は自動的に計算
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

        # アテンションの重みを格納するために使われる max_length を計算
        max_length = self.calc_max_length(train_seqs)
        return cap_vector, max_length