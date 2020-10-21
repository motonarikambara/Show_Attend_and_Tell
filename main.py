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
from tqdm import tqdm
import datetime

import reg_bleu_score as bleu_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from caption_generator import Caption_Generator
import data_loader

class ShowAttendandTell():
    def __init__(self):
        self.captiongenerator = Caption_Generator() 
        self.tokenizer = self.captiongenerator.tokenizer

    def train(self):
        self.ans = []
        self.res = 0
        self.score = {'BLEU4':0, 'ROUGE':0, 'METEOR':0, 'CiDEr':0}
        print("Start trining")
        ckpt_manager, self.dataset = self.captiongenerator.model()
        start_epoch = 0
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

        start_epoch = 0
        EPOCHS = 1
        for epoch in range(start_epoch, EPOCHS):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(tqdm(self.dataset)):
                batch_loss, t_loss = self.captiongenerator.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

            # if epoch % 5 == 0:
            #     ckpt_manager.save()

            print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                                total_loss/self.captiongenerator.num_steps))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            print("Start validation")
            self.test(self.captiongenerator.img_name_val, self.captiongenerator.cap_val)
            print("End validation")
            if self.res < self.me:
                self.res = self.me
                ckpt_manager.save()
                print("Start test")
                self.test(self.captiongenerator.img_name_test, self.captiongenerator.cap_test, test=True)
                print("End test")

        print("End training")
        now = datetime.datetime.now()
        path_cap = 'result/{0:%Y%m%d}_cap.txt'.format(now)
        path_img = 'result/{0:%Y%m%d}_img.txt'.format(now)
        with open(path_cap, mode='w') as f:
            f.write('\n'.join(self.ans))
        with open(path_img, mode='w') as f:
            f.write('\n'.join(self.captiongenerator.img_name_test))
        print("BLEU4: ", self.score['BLEU4'])
        print("ROUGE: ", self.score['ROUGE'])
        print("METEOR: ", self.score['METEOR'])
        print("CiDEr: ", self.score['CiDEr'])

    def test(self, img, caption, test=False):
        print("Start test")
        # 検証用セットのキャプション
        # results = list()
        # for i in tqdm(range(len(img))):
        #     image = img[i]
        #     result = self.captiongenerator.evaluate(image)

        #     g_sent = ' '.join(result)
        #     res = g_sent.split('.')[0]
        #     final_sent = res.split('?')[0].rstrip()
        #     results.append(final_sent)
        ans = caption       
        for i in range(len(ans)):
            ans[i] = ans[i].rstrip()
        
        ########################################################
        # metric test
        # 実行時には消すこと
        results = self.captiongenerator.cheat
        for i in range(len(results)):
            results[i] = results[i].rstrip()
        # for i in range(len(ans)):
        #     ans[i] = ans[i].rstrip()
        #########################################################


        reference_dict = dict()
        generated_dict = dict()
        if not test:
            target_id = self.captiongenerator.val_target
            for iid, tid, ref, gen in zip(img, target_id, ans, results):
                    reference_dict.setdefault(str(iid).zfill(
                        4) + "_" + str(tid).zfill(4), []).append(self.metric_preprocess([ref])[0])
                    generated_dict[str(iid).zfill(
                        4) + "_" + str(tid).zfill(4)] = self.metric_preprocess([gen])
        else:
            target_id = self.captiongenerator.test_target
            for iid, tid, ref, gen in zip(img, target_id, ans, results):
                    reference_dict.setdefault(str(iid).zfill(
                        4) + "_" + str(tid).zfill(4), []).append(self.metric_preprocess([ref])[0])
                    generated_dict[str(iid).zfill(
                        4) + "_" + str(tid).zfill(4)] = self.metric_preprocess([gen])
        # Make dict to calcurate metrics
        # print(reference_dict)

        total_score, sc_bl, sc_rg, sc_me, sc_cid = self.calculate_metric_scores(reference_dict, generated_dict)
        print("BLEU4: %.3f" % sc_bl[3])
        print("ROUGE: %.3f" % sc_rg)
        print("METEOR: %.3f" % sc_me)
        print("CiDEr: %.3f" % sc_cid)
        self.me = sc_me
        if test:
            self.score['BLEU4'] = sc_bl
            self.score['ROUGE'] = sc_rg
            self.score['METEOR'] = sc_me
            self.score['CiDEr'] = sc_cid
            self.ans = results

    
    # 入力：正解の辞書，生成文の辞書
    # 出力：各スコアの合計及び各スコア
    def calculate_metric_scores(self, references, candidates):

        metric_start = time.time()
        ###BLEU#####
        pycoco_bleu = Bleu()
        sc_bl, _ = pycoco_bleu.compute_score(references, candidates)
        bleu_end = time.time()

        ####METEOR###
        pycoco_meteor = Meteor()
        sc_me, _ = pycoco_meteor.compute_score(
            references, candidates)
        meteor_end = time.time()
        del pycoco_meteor

        ####ROUGE###
        pycoco_rouge = Rouge()
        sc_rg, _ = pycoco_rouge.compute_score(
            references, candidates)
        rouge_end = time.time()

        ####CIDER###
        pycoco_cider = Cider()
        sc_cid, _ = pycoco_cider.compute_score(
            references, candidates)
        cider_end = time.time()

        # print(
        #     "Metric times: bleu[{0:.2f}] meteor[{1:.2f}] rouge[{2:.2f}] cider[{3:.2f}]".format(
        #         bleu_end-metric_start, meteor_end-bleu_end, rouge_end-meteor_end, cider_end-rouge_end))

        # sc_me = 1.0

        total_score = sc_bl[0] + sc_bl[1] + \
            sc_bl[2] + sc_bl[3] + sc_rg + sc_me + sc_cid

        return total_score, sc_bl, sc_rg, sc_me, sc_cid
    
    # 入力：トークンの入ったリスト
    # 出力：記号をなくしたリスト
    def metric_preprocess(self, sentence_list):
        new_sentence_list = list()
        # print(sentence_list)
        for s in sentence_list:
            s_new = s.replace(".", "").replace(" ##", "").replace("?", "").replace("!", "").replace("<", "").replace(">", "").replace("\n", "").lower()
            new_sentence_list.append(s_new)
        return new_sentence_list

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    model = ShowAttendandTell()
    model.train()
    # model.test(model.captiongenerator.img_name_test, model.captiongenerator.cap_test)
