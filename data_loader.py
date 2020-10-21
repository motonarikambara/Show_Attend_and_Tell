import tensorflow as tf

# Scikit-learn には役に立つさまざまなユーティリティが含まれる
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm
import csv
import caption_generator

class DataLoader():
    def __init__(self):
        # annotation_zip = tf.keras.utils.get_file('captions.zip',
        #                                         cache_subdir=os.path.abspath('.'),
        #                                         origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        #                                         extract = True)
        # annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

        # name_of_zip = 'train2014.zip'
        # if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        #     image_zip = tf.keras.utils.get_file(name_of_zip,
        #                                         cache_subdir=os.path.abspath('.'),
        #                                         origin = 'http://images.cocodataset.org/zips/train2014.zip',
        #                                         extract = True)
        #     PATH = os.path.dirname(image_zip)+'/train2014/'
        # else:
        #     PATH = os.path.abspath('.')+'/train2014/'
        annotation_dir = os.path.abspath('.') + '/wsr20/labels/'
        train_annot = os.path.dirname(annotation_dir) + '/sentences_train_wrs.csv'
        train_imgs = os.path.dirname(annotation_dir) + '/image_id_train.csv'
        val_annot = os.path.dirname(annotation_dir) + '/sentences_val_wrs.csv'
        val_imgs = os.path.dirname(annotation_dir) + '/image_id_val.csv'
        test_annot = os.path.dirname(annotation_dir) + '/sentences_test_wrs.csv'
        test_imgs = os.path.dirname(annotation_dir) + '/image_id_test.csv'
        test_target = os.path.dirname(annotation_dir) + '/target_id_test.csv'
        val_target = os.path.dirname(annotation_dir) + '/target_id_val.csv'


        trainannotations = []
        trainimgs = []
        valannotations = []
        valimgs = []
        testannotations = []
        testimgs = []
        targetid = []
        valtargetid= []
        # csvファイルの読み込み
        with open(train_annot, 'r') as f:
            tmp = csv.reader(f)
            for i in f:
                trainannotations.append(i)
        with open(train_imgs, 'r') as f:
            tmp = csv.reader(f)
            for i in f:
                trainimgs.append(i)
        with open(val_annot, 'r') as f:
            tmp = csv.reader(f)
            for i in f:
                valannotations.append(i)
        with open(val_imgs, 'r') as f:
            tmp = csv.reader(f)
            for i in f:
                valimgs.append(i)
        with open(test_annot, 'r') as f:
            tmp = csv.reader(f)
            for i in f:
                testannotations.append(i)
        with open(test_imgs, 'r') as f:
            tmp = csv.reader(f)
            for i in f:
                testimgs.append(i)
        with open(test_target, 'r') as f:
            tmp = csv.reader(f)
            for i in f:
                targetid.append(i)
        with open(val_target, 'r') as f:
            tmp = csv.reader(f)
            for i in f:
                valtargetid.append(i)
        #############################################################
        tmptestannotations = []
        tmptest_annot = os.path.abspath('.') + '/generated_sentence_test000.txt'
        with open(tmptest_annot, 'r') as f:
            for i in f:
                tmptestannotations.append(i)
        ###############################################################

        PATH = os.path.abspath('.') + '/wsr20/single_view_img_data/'
        # json ファイルの読み込み
        # with open(annotation_file, 'r') as f:
        #     annotations = json.load(f)

        # image id & caption
        tr_caption = []
        tr_imgs = []
        va_caption = []
        va_imgs = []
        tes_caption = []
        tes_imgs = []
        tmptes_caption = []

        for i in range(len(trainannotations)):
            caption = '<start> ' + trainannotations[i] + ' <end>'
            tr_caption.append(caption)
            image_path = PATH + '%04d.jpg' % (int(trainimgs[i]))
            tr_imgs.append(image_path)

        for i in range(len(valannotations)):
            caption = '<start> ' + valannotations[i] + ' <end>'
            va_caption.append(caption)
            image_path = PATH + '%04d.jpg' % (int(valimgs[i]))
            va_imgs.append(image_path)
        
        for i in range(len(testannotations)):
            # caption = '<start> ' + testannotations[i] + ' <end>'
            caption = testannotations[i]
            tes_caption.append(caption)
            image_path = PATH + '%04d.jpg' % (int(testimgs[i]))
            tes_imgs.append(image_path)
        
        for i in range(len(tmptestannotations)):
            caption = tmptestannotations[i]
            tmptes_caption.append(caption)

        self.cheat = tmptes_caption

        self.train_captions, self.train_images = shuffle(tr_caption,
                                                tr_imgs,
                                                random_state=1)
        self.val_captions, self.val_images = shuffle(va_caption,
                                                va_imgs,
                                                random_state=1)
        self.test_captions = tes_caption
        self.test_images = tes_imgs
        self.test_target = targetid
        self.val_target = valtargetid
                                 
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    # numpy ファイルをロード
    def map_func(self, img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        # img_tensor = np.array(img_name.decode('utf-8'))
        # img_tensor = np.array(Image.open(img_name.decode('utf-8')))
        return img_tensor, cap
    
    def dataprepare(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output

        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        # 重複のない画像を取得
        encode_train = sorted(set(self.train_images))

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

        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'

        # トークン化したベクトルを生成
        train_seqs = self.tokenizer.texts_to_sequences(self.train_captions)

        # キャプションの最大長に各ベクトルをパディング
        # max_length を指定しない場合、pad_sequences は自動的に計算
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

        # アテンションの重みを格納するために使われる max_length を計算
        max_length = self.calc_max_length(train_seqs)
        return cap_vector, max_length