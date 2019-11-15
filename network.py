

import tensorflow as tf
import numpy as np
from environment import Environment
import tensorflow.contrib.slim as slim
import random
from agent import Agent

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer

class Qnetwork:
    def __init__(self, learning_rate = 0.01, model_type = 1, name=None):
        with tf.variable_scope(name):
            self.model_type = model_type

            #매개변수 : 입력값 크기(이미지), 출력개수, 학습률
            self.inImage = tf.placeholder(tf.float32,shape=[None]+Environment.RANGE_SHAPE[model_type])
            self.phase= tf.placeholder(tf.bool, name='phase')

            self.temp = tf.placeholder(tf.float32,shape=None)
            self.keep_per = tf.placeholder(tf.float32,shape=None)

            self.portfolio_state = tf.placeholder(tf.float32, shape=[None,Agent.STATE_DIM])

            conv1 = tf.contrib.layers.convolution2d(inputs=self.inImage, num_outputs=32, kernel_size=[8, 8],
                                                    padding="VALID", biases_initializer=None)
            mp1 = tf.contrib.layers.max_pool2d(conv1, kernel_size=[3, 3])

            conv2 = tf.contrib.layers.convolution2d(inputs=mp1, num_outputs=32, kernel_size=[8, 8],
                                                    padding="VALID", biases_initializer=None)
            mp2 = tf.contrib.layers.max_pool2d(conv2, kernel_size=[3, 3])

            conv3 = tf.contrib.layers.convolution2d(inputs=mp2, num_outputs=32, kernel_size=[8, 8],
                                                    padding="VALID", biases_initializer=None)
            mp3 = tf.contrib.layers.max_pool2d(conv3, kernel_size=[3, 3])
            conv4 = tf.contrib.layers.convolution2d(inputs=mp3, num_outputs=32, kernel_size=[8, 8],
                                                    padding="VALID", biases_initializer=None)
            mp4 = tf.contrib.layers.max_pool2d(conv4, kernel_size=[3, 3])

            #img_result = tf.reshape(mp4, [-1, mp4.get_shape().as_list()[0]])
            img_result = tf.layers.flatten(mp4)

            # 하나의 LSTML셀에 몇개의 뉴럴을 생성할 것인지 설정
            lstm_cell = tf.contrib.rnn.LSTMCell(1024, state_is_tuple=True)

            # 게임에 사용하기전 상태값 초기화용도
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)

            self.state_init = [c_init, h_init]

            # 게임중에 입력으로 넣을 셀,히든 상태값
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c],name='c_state')
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h],name='h_state')

            self.state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(img_result, [0])
            # LSTM 셀 길이를 지정하기 위해 입력 행개수를 시퀀스 랭스 매개변수에 입력
            step_size = tf.shape(img_result)
            step_size = step_size[0:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in,
                                                         sequence_length=step_size, time_major=False)

            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out =  tf.reshape(lstm_outputs, [-1, 1024])

            batch_norm = tf.contrib.layers.batch_norm(rnn_out, is_training=self.phase, scope='bn', center=True,
                                                      scale=True)

            concat = tf.concat([batch_norm,self.portfolio_state], 1)

            fc_layer_1 = slim.fully_connected(concat, 512, activation_fn=tf.nn.leaky_relu,
                                              weights_initializer=normalized_columns_initializer(), biases_initializer=None)

            dropout = slim.dropout(fc_layer_1, self.keep_per)

            self.streamAC, self.streamVC = tf.split(dropout, [256, 256], 1)

            self.streamA = tf.contrib.layers.flatten(self.streamAC)
            self.streamV = tf.contrib.layers.flatten(self.streamVC)

            self.AW = tf.Variable(tf.random_normal([256, Agent.NUM_ACTIONS]))
            self.VW = tf.Variable(tf.random_normal([256, 1]))

            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)
            # Q = V + A<-(A - A평균)
            #self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
            # Q = V + A<-(A - Amax)
            self.Qout = self.Value + tf.subtract(self.Advantage,
                                                 tf.reduce_max(self.Advantage, reduction_indices=1, keep_dims=True))
            self.predict = tf.argmax(self.Qout, 1)
            self.Q_dist = tf.nn.softmax(self.Qout/self.temp)#(self.Qout/self.temp)


            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

            self.actions_onehot = tf.one_hot(self.actions, depth=Agent.NUM_ACTIONS)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)

            self.td_error = tf.losses.huber_loss(labels=self.targetQ,predictions=self.Q)
            #self.td_error = tf.square(self.targetQ - self.Q)

            self.loss = tf.reduce_mean(self.td_error)

            self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            self.updateModel = self.trainer.minimize(self.loss)
import sys
class experience_buffer():
    def __init__(self, buffer_size=140):
        self.buffer = []
        self.reward_buffer=[]
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            #self.buffer[0:1] = []
            del self.buffer[0]

        self.buffer.append(experience)

        #print(sys.getsizeof(self.buffer))

    def sample(self, size, reward_list = None):
        if reward_list:
            reward_list = np.array(reward_list,dtype=np.float32)
            reward_list += abs(np.min(reward_list))+1
            sum_r =np.sum(reward_list)
            reward_list /= sum_r
            idx = np.random.choice(len(self.buffer),1,p=reward_list)
            episode = self.buffer[idx[0]]
        else:
        #return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
            episode = random.sample(self.buffer, 1)[0]
        size = min(len(episode), size)
        buffer = random.randint(0, len(episode) - size)
        result = np.array(episode[buffer: buffer + size]), size
        return result

def updateTargetGraph(tfVars, tau):
    # tfVars 는 학습 가능한 변수들
    # tau는 타겟 신경망이 학습 신경망을 향하는 비율
    # 학습 가능한 변수들의 수
    total_vars = len(tfVars)

    op_holder = []

    for idx, var in enumerate(tfVars[0:int(total_vars / 2)]):
        op_holder.append(tfVars[int(idx) + int(total_vars / 2)].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[int(idx) + int(total_vars / 2)].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


