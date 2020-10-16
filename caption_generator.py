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

from data_loader import DataLoader

class Caption_Generator():
    def __init__(self):
        self.dataloader = DataLoader()
        self.dataloader.dataprepare()
        self.cap_vector, self.max_length = self.dataloader.tokenize()
        self.tokenizer = self.dataloader.tokenizer


    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    
    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0

        # バッチごとに隠れ状態を初期化
        # 画像のキャプションはその前後の画像と無関係なため
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # 特徴量をデコーダに渡す
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
                # teacher forcing を使用
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss, total_loss

    def evaluate(self, image):

        hidden = self.decoder.reset_state(batch_size=1)
        temp_input = tf.expand_dims(self.dataloader.load_image(image)[0], 0)
        img_tensor_val = self.dataloader.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            if predicted_id not in self.tokenizer.index_word:
                return result
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result

            dec_input = tf.expand_dims([predicted_id], 0)
        # print(result)
        return result

    def model(self):
        # これらのパラメータはシステム構成に合わせて自由に変更してください
        # img_name_train, self.img_name_val, cap_train, self.cap_val = train_test_split(self.dataloader.img_name_vector,
        #                                                                         self.cap_vector,
        #                                                                         test_size=0.2,
        #                                                                         random_state=0)
        img_name_train = self.dataloader.train_img
        self.img_name_val = self.dataloader.test_img
        cap_train = self.cap_vector
        self.cap_val = self.dataloader.test_cap
        BATCH_SIZE = 64
        BUFFER_SIZE = 1000
        embedding_dim = 256
        units = 512
        vocab_size = len(self.tokenizer.word_index) + 1
        self.num_steps = len(img_name_train) // BATCH_SIZE
        # InceptionV3 から抽出したベクトルの shape は (64, 2048)
        # つぎの 2 つのパラメータはこのベクトルの shape を表す
        features_shape = 2048
        attention_features_shape = 64

        dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

        # numpy ファイルを並列に読み込むために map を使用
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
                self.dataloader.map_func, [item1, item2], [tf.float32, tf.int32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # シャッフルとバッチ化
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        checkpoint_path = "./checkpoints/train"
        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                decoder=self.decoder,
                                optimizer = self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        return ckpt_manager, dataset

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, 64, 1)
        # score を self.V に適用するので、最後の軸は 1 となる
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # 合計をとったあとの　context_vector の shpae == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # すでに特徴量を抽出して pickle 形式でダンプしてあるので
    # このエンコーダはそれらの特徴量を全結合層に渡して処理する
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # アテンションを別のモデルとして定義
        context_vector, attention_weights = self.attention(features, hidden)
        # embedding 層を通過したあとの x の shape == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # 結合後の x の shape == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # 結合したベクトルを GRU に渡す
        output, state = self.gru(x)
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)
        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
