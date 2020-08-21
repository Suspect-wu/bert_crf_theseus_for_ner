#! -*- coding:utf-8 -*-

import os
os.environ['TF_KERAS'] = '1'

import tensorflow as tf
from bert4keras.backend import K, batch_gather, keras
import pandas as pd
from bert4keras.layers import Dense, ConditionalRandomField, Input
from tensorflow.keras.models import Model
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import *
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from tqdm import tqdm
from bert4keras.snippets import to_array, ViterbiDecoder

config = 'roberta_zh_l12/bert_config.json'
ckpt = 'roberta_zh_l12/bert_model.ckpt'

dict_path = 'roberta_zh_l12/vocab.txt'
train_path = 'data/example.train'
dev_path = 'data/example.dev'

epochs = 5
maxlen = 256
num_labels = 7
batch_size = 32

# 加载数据
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        # 按一整个句子划分
        for l in f.split('\n\n'):
            if not l:
                continue
            # d是对每个句子
            d, last_flag = [], ''
            flag = 0
            for c in l.split('\n'):
                try:
                    char, this_flag = c.split(' ')
                # 当出现了 '  I-ORG' 出现空格时，会报错
                except:
                    flag = 1
                    break
                    # pass
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            # 整个数据集
            if not flag:
                D.append(d)
    # 输出D[[['吕', 'PER'], ['老客厅中仅有三个几十年前的旧木沙发和一张早就掉了漆的木头八仙桌，这可能同记者想象中蜚声国内外的大学者家里的陈设太不相称了。', 'O']], [[], []]]
    return D

class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        # is_end 判断数据集是否到头
        # items [['乔庄', 'LOC'], ['的村民们都怕自己跟“穷”字沾边儿，为了早一天过上好日子，他们在默默地做着各自的打算。', 'O']]
        for is_end, item in self.sample(random):
            # token_ids 句子开头的标签[101]
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                # 转换成词标签，不要开头的[cls]和seg
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                # 这一整个句子长度保持在maxlen之下
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    # w代表小短句， l代表该句的标签
                    # labels 该句子的标签
                    # w_token_ids = [102, 4584, ... , ]
                    # labels = [0, 1(B-LOC), 2(I-LOC), ... , 0]
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                # 截断句子
                else:
                    break
            # 这一整个句子，加上结尾的标签
            # [cls]开头 102. [sep]结尾101
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            # 加入该batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            # 当batch足够或者到了数据结尾，返回结果
            if len(batch_token_ids) == self.batch_size or is_end:
                # 长度不够的话，补上0
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                # [batch_size, seq]
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text, model):
        # tokens
        # ’第一章：前言	4‘ --> ['[CLS]', '第', '一', '章', '：', '前', '言', '4', '[SEP]']
        # 去掉空格
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)

        # tokens = ['[CLS]', '第', '一', '章', '：', '前', '言', '4', '[SEP]'], text = ’第一章：前言	4‘
        # [[], [0], [1], [2], [3], [4], [5], [7], []]
        # mapping为tokens中的字符在text中对应的索引
        mapping = tokenizer.rematch(text, tokens)

        # token_ids
        # ['[CLS]'， '[SEP]'] --> [101, 102]
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        # 转换为矩阵
        token_ids, segment_ids = to_array([token_ids], [segment_ids])

        # predict 输出三维，[batch_size, seq, K]
        # 因为batch = 1， nodes[0] 变成二位

        nodes = model.predict([token_ids, segment_ids])[0]

        # 输出最优路径的标签[0,0,0,1,2,2,2,0,0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True

                    # i为输入seq的索引,之后需要通过mapping对应到text中的索引
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        # entity[ [[1, 2, 3]， 'ORG'], [[7, 8]， 'ORG']] ]
        # 英文的时候会需要mapping[w[0]][0]和mapping[w[-1]][-1]
        #
        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


def evaluate(data, model):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    # d 代表每一句 [['乔庄', 'LOC'], ['的村民们都怕自己跟“穷”字沾边儿，为了早一天过上好日子，他们在默默地做着各自的打算。', 'O']]
    for d in tqdm(data):
        # text str
        text = ''.join([i[0] for i in d])
        # 输出
        R = set(NER.recognize(text, model=model))
        #
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluate(keras.callbacks.Callback):
    def __init__(self, save_model_path, valid_data):
        self.best_val_f1 = 0
        self.save_model_path = save_model_path
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        # 这个时候，会更新状态转移矩阵
        trans = K.eval(crf.trans)
        NER.trans = trans
        print('\n 状态转移矩阵：')
        print(NER.trans)
        f1, precision, recall = evaluate(self.valid_data, self.model)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.save_model_path, save_format='HDF5')
        # 打印验证集
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


def lr_schedual(epoch):
    if epoch < 4:
        return 2e-5
    else:
        return 3e-6


class BinaryRandomChoice(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BinaryRandomChoice, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[1]

    # 训练的时候，随机选择前辈和后辈的输出
    # 测试的时候，选择后辈的输出
    def call(self, inputs, **kwargs):
        source, target = inputs
        mask = K.random_binomial(shape=[1], p=0.5)
        # 1 选择前辈
        # 0 选择后辈
        output = mask * source + (1- mask) * target
        return K.in_train_phase(output, target)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

def bert_of_theseus(predecessor, successor, classfier):
    inputs = predecessor.inputs
    for layer in predecessor.model.layers:
        layer.trainable = False
    classfier.trainable = False

    # 替换embedding层
    predecessor_outputs = predecessor.apply_embeddings(inputs)
    successor_outputs = successor.apply_embeddings(inputs)

    outputs = BinaryRandomChoice(name='Embedding_choice')([predecessor_outputs, successor_outputs])

    # 替换Transformer层

    layer_per_module = predecessor.num_hidden_layers // successor.num_hidden_layers

    for index in range(successor.num_hidden_layers):
        predecessor_outputs = outputs
        for sub_index in range(layer_per_module):
            predecessor_outputs = predecessor.apply_main_layers(
                predecessor_outputs, layer_per_module * index + sub_index
            )
        successor_outputs = successor.apply_main_layers(outputs, index)
        outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])

    outputs = classfier([outputs])
    model = Model(inputs, outputs, name='theseus')
    return model

# 判别模型
def Classfier():

    x_in = Input(shape=(None, 768), name='bert_output')

    output = Dense(units=num_labels)(x_in)

    crf = ConditionalRandomField(lr_multiplier=1000, name='CRF')

    output = crf(output)

    classfier = tf.keras.Model(x_in, output, name='classfier')

    classfier.summary()

    return classfier, crf


def Bert_layer(num_hidden_layers, prefix):
    bert = build_transformer_model(
                                   config_path=config,
                                   checkpoint_path=ckpt,
                                   return_keras_model=False,
                                   num_hidden_layers=num_hidden_layers,
                                   prefix=prefix
                                   )
    return bert

def All_model():
    classfier, crf = Classfier()

    # -------------------------前辈-----------------------------#
    predecessor_bert = Bert_layer(num_hidden_layers=12, prefix='Predecessor-')

    predecessor = tf.keras.Model(
        predecessor_bert.inputs,
        classfier(predecessor_bert.outputs),
        name='Predecessor'
    )

    predecessor.summary()

    predecessor.compile(
        loss=crf.sparse_loss,
        optimizer=Adam(1e-5),
        metrics=[crf.sparse_accuracy]
    )

 # ---------------------------后辈 ------------------------- #
    successor_bert = Bert_layer(num_hidden_layers=3, prefix='Successor-')

    successor = tf.keras.Model(
        successor_bert.inputs,
        classfier(successor_bert.outputs),
        name='successor'
    )
    successor.compile(
        loss=crf.sparse_loss,
        optimizer=Adam(1e-5),
        metrics=[crf.sparse_accuracy]
    )
    successor.summary()

    #---------------------------忒休斯--------------------------#

    theseus = bert_of_theseus(predecessor_bert, successor_bert, classfier)
    theseus.summary()
    theseus.compile(
        loss=crf.sparse_loss,
        optimizer=Adam(1e-5),
        metrics=[crf.sparse_accuracy]
    )

    return predecessor, successor, theseus, crf


if __name__ == "__main__":
    # 标注数据
    train_data = load_data(train_path)
    valid_data = load_data(dev_path)
    # train_data = train_data[:2]
    # valid_data = valid_data[:2]
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    # 类别映射
    labels = ['ORG', 'PER', 'LOC']
    id2label = dict(enumerate(labels))
    label2id = {j: i for i, j in id2label.items()}
    # 5个类别
    num_labels = len(labels) * 2 + 1

    predecessor, successor, theseus, crf = All_model()
    # K.eval将变量转换为矩阵
    NER = NamedEntityRecognizer(trans=K.eval(crf.trans), starts=[0], ends=[0])

    train_generator = data_generator(train_data, batch_size)

    print('----------------------------训练先辈--------------------------')

    predecessor_evaluator = Evaluate(
        save_model_path='./model_weights/best_predecessor.weights',
        valid_data=valid_data)

    predecessor.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[predecessor_evaluator]
    )

    # 训练忒休斯，后辈的bert效果接近先辈的bert
    print('----------------------------训练忒休斯--------------------------')

    theseus_evaluator = Evaluate(
        save_model_path='./model_weights/best_theseus.weights',
        valid_data=valid_data)

    theseus.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=2*epochs,
        callbacks=[theseus_evaluator]
    )

    # 微调后辈
    print('----------------------------微调后辈--------------------------')

    successor_evaluator = Evaluate(
        save_model_path='./model_weights/best_successor.weights',
        valid_data=valid_data)

    successor.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[successor_evaluator]
    )