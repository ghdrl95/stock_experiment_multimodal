

import tensorflow as tf
import numpy as np
from environment import Environment
import tensorflow.contrib.slim as slim
import random
from agent import Agent
from copy import deepcopy
import PIL.Image as pilimg

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
class TrendPredictNetwork:

    MODEL_PATH = ['./stock_predict_basic','./stock_predict_bb','./stock_predict_macd','./stock_predict_obv','./stock_predict_dmi','./stock_predict_sto']
    INPUT_IMAGE_PATH = ['C:/chart_images_other_basic/','C:/chart_images_other_BB/','C:/chart_images_other_MACD/','C:/chart_images_other_OBV/','C:/chart_images_other_DMI/','C:/chart_images_other_STO/']
    LSTM_LAYERS = 4
    DEVICES=['/cpu:0','/gpu:0']

    TYPE_BASIC = 0  # 캔들차트 + 이평선 + 거래량
    TYPE_BB = 1     # 볼린저 밴드
    TYPE_MACD = 2   # MACD
    TYPE_OBV = 3    # OBV
    TYPE_DMI = 4    # DMI
    TYPE_STO = 5    # 스토캐스팅

    #RANGE_SHAPE = {5: [630, 130, 4], 20: [630, 245, 4], 60: [630, 550, 4], 6:[630,130,4], 7:[630,130,4], 25: [630, 280, 4]}
    #RANGE_SHAPE = {20: [400, 240, 4], 5: [630, 130, 4],  60: [630, 550, 4], 6: [630, 130, 4], 7: [630, 130, 4],25: [630, 280, 4]}
    RANGE_SHAPE = { 20 : [ [400, 240, 4], [320,235,4],[320,245,4],[320,250,4],[320,215,4],[320,215,4] ] }

    def __init__(self, model_type, is_train = True, data_type = 0, learning_rate = 0.01, name = None):
        # 신경망 구성
        self.model_type = model_type
        self.data_type = data_type
        device = self.DEVICES[1] if is_train else self.DEVICES[0]
        #self.SEQ_LENGTH = self.SEQ_LENGTH if is_train else 1
        with tf.device(device):
            with tf.variable_scope(name):
                self.phase = tf.placeholder(tf.bool, name='phase')
                self.in_batch_size = tf.placeholder(tf.int32)
                self.temp = tf.placeholder(tf.float32, shape=None)
                self.keep_per = tf.placeholder(tf.float32, shape=None)
                self.portfolio_state = tf.placeholder(tf.float32, shape=[None, Agent.STATE_DIM])
            batch_norm_list = []
            self.inImage = []
            for i, type in enumerate(self.data_type):
                with tf.variable_scope(name+str(type)):
                    # 매개변수 : 입력값 크기(이미지), 출력개수, 학습률
                    self.inImage.append(tf.placeholder(tf.float32, shape=[None] + Environment.RANGE_SHAPES[model_type][type]))

                    #?,778,243,32
                    conv1 = tf.contrib.layers.convolution2d(inputs=self.inImage[i], num_outputs=32, kernel_size=[14, 14],
                                                            padding="VALID", biases_initializer=None,trainable=is_train)
                    batch_norm_1 = tf.contrib.layers.batch_norm(conv1, is_training=self.phase, scope='bn_1', center=True, scale=True)
                    #?,388,121,32
                    mp1 = tf.contrib.layers.max_pool2d(batch_norm_1, kernel_size=[3, 3],stride=3)
                    # ?,386,119,32
                    conv2 = tf.contrib.layers.convolution2d(inputs=mp1, num_outputs=32, kernel_size=[7, 7],
                                                            padding="VALID", biases_initializer=None,trainable=is_train)
                    batch_norm_2 = tf.contrib.layers.batch_norm(conv2, is_training=self.phase, scope='bn_2', center=True,
                                                                scale=True)
                    # ?,192,59,32
                    mp2 = tf.contrib.layers.max_pool2d(batch_norm_2, kernel_size=[3, 3])

                    conv3 = tf.contrib.layers.convolution2d(inputs=mp2, num_outputs=32, kernel_size=[3, 3],
                                                            padding="VALID", biases_initializer=None,trainable=is_train)
                    batch_norm_3 = tf.contrib.layers.batch_norm(conv3, is_training=self.phase, scope='bn_3', center=True,
                                                                scale=True)
                    mp3 = tf.contrib.layers.max_pool2d(batch_norm_3, kernel_size=[3, 3])
                    conv4 = tf.contrib.layers.convolution2d(inputs=mp3, num_outputs=32, kernel_size=[3, 3],
                                                            padding="VALID", biases_initializer=None,trainable=is_train)
                    batch_norm_4 = tf.contrib.layers.batch_norm(conv4, is_training=self.phase, scope='bn_4', center=True,
                                                                scale=True)
                    mp4 = tf.contrib.layers.max_pool2d(batch_norm_4, kernel_size=[3, 3])

                    conv5 = tf.contrib.layers.convolution2d(inputs=mp4, num_outputs=64, kernel_size=[3, 3],
                                                            padding="VALID", biases_initializer=None,trainable=is_train)
                    batch_norm_5 = tf.contrib.layers.batch_norm(conv5, is_training=self.phase, scope='bn_5', center=True,
                                                                scale=True)

                    # img_result = tf.reshape(mp4, [-1, mp4.get_shape().as_list()[0]])
                    #img_result = tf.layers.flatten(batch_norm_5)
                    img_result = tf.reshape( tf.transpose(batch_norm_5,perm=[0,2,1,3]),shape=[-1,batch_norm_5.shape[2],batch_norm_5.shape[1]*batch_norm_5.shape[3]])

                    # 하나의 LSTML셀에 몇개의 뉴럴을 생성할 것인지 설정
                    lstms=[]
                    for i in range(self.LSTM_LAYERS):
                        lstm_cell = tf.contrib.rnn.LSTMCell(1024, state_is_tuple=True,trainable=is_train,name="LSTM_%d"%i)
                        lstms.append(lstm_cell)
                    cell = tf.contrib.rnn.MultiRNNCell(lstms)

                    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(cell, img_result, dtype=tf.float32)

                    #rnn_out = tf.reshape(lstm_outputs, [-1, 1024])[-1:]
                    rnn_out = lstm_outputs[:,-1]

                    batch_norm = tf.contrib.layers.batch_norm(rnn_out, is_training=self.phase, scope='bn', center=True,
                                                              scale=True)
                    batch_norm_list.append(batch_norm)
            with tf.variable_scope(name):
                concat = tf.concat([norm for norm in batch_norm_list]+[self.portfolio_state], 1)

                fc_layer_1 = slim.fully_connected(concat, 512, activation_fn=tf.nn.leaky_relu,
                                                  weights_initializer=normalized_columns_initializer(),
                                                  biases_initializer=None)

                dropout = slim.dropout(fc_layer_1, self.keep_per)

                self.streamAC, self.streamVC = tf.split(dropout, [256, 256], 1)

                self.streamA = tf.contrib.layers.flatten(self.streamAC)
                self.streamV = tf.contrib.layers.flatten(self.streamVC)

                self.AW = tf.Variable(tf.random_normal([256, Agent.NUM_ACTIONS]))
                self.VW = tf.Variable(tf.random_normal([256, 1]))

                self.Advantage = tf.matmul(self.streamA, self.AW)
                self.Value = tf.matmul(self.streamV, self.VW)
                # Q = V + A<-(A - A평균)
                # self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
                # Q = V + A<-(A - Amax)
                self.Qout = self.Value + tf.subtract(self.Advantage,
                                                     tf.reduce_max(self.Advantage, reduction_indices=1, keep_dims=True))
                self.predict = tf.argmax(self.Qout, 1)
                self.Q_dist = tf.nn.softmax(self.Qout / self.temp)  # (self.Qout/self.temp)

                self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

                self.actions_onehot = tf.one_hot(self.actions, depth=Agent.NUM_ACTIONS)

                self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)

                self.td_error = tf.losses.huber_loss(labels=self.targetQ, predictions=self.Q)
                # self.td_error = tf.square(self.targetQ - self.Q)

                self.loss = tf.reduce_mean(self.td_error)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name)
                with tf.control_dependencies(update_ops):
                    self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    self.updateModel = self.trainer.minimize(self.loss)

    def append_chartdata_in_batch(self, list, x_buffer, y_buffer, idx_buffer, last_idx_buffer, code_buffer, y_list, is_train = True):
        if len(list) > 0:
            chart_code=''
            while len(list) > 0:
                chart_code = list[0]
                del list[0]
                if self.chartcode_value[chart_code] >= 0:
                    try:
                        chart_data_y = np.genfromtxt(self.OUTPUT_DATA_PATH + chart_code + ".csv", delimiter=',')
                    except OSError:
                        chart_code = ''
                        continue
                    break
                else:
                    self.chartcode_list.remove(chart_code)
                    chart_code = ''
            if chart_code == "":
                self.has_data = False
                return
            if is_train:
                last_idx = 180 + int((chart_data_y.shape[0]) * 0.8)
                idx = 180
            else:
                idx = 180 + int((chart_data_y.shape[0]) * 0.8)
                last_idx = 179 + chart_data_y.shape[0]
            print('buffer load, code : %s, idx : %s, last_idx : %s reamining data : %s' % (chart_code, idx, last_idx,len(list)))

            input_x = self.get_data(idx, chart_code)
            if input_x is not None:
                input_x = [input_x]
            else:
                print('데이터 없음')
                return None

            x_buffer.extend(input_x)
            y_buffer.append(chart_data_y)
            idx_buffer.append(idx)
            last_idx_buffer.append(last_idx)
            code_buffer.append(chart_code)
            y_list.append(chart_data_y[np.where(chart_data_y == idx)[0][0], 1])
        else:
            self.has_data = False

    def train(self,isTrain = False):

        #config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=1,reshape=True)
            if isTrain:
                print('Loading Model')
                ckpt = tf.train.get_checkpoint_state(self.MODEL_PATH[self.data_type])
                saver.restore(sess, ckpt.model_checkpoint_path)
            for epoch in range(1, TrendPredictNetwork.EPOCH_NUM + 1):
                print('EPOCH %s start' % epoch)
                list = deepcopy(self.chartcode_list)
                random.shuffle(list)
                list=list[:len(list)//self.DIVIVE_DATA]
                x_buffer = []
                y_buffer= []

                idx_buffer= []
                last_idx_buffer= []
                code_buffer= []
                self.has_data = True
                cnt = 0
                while True:
                    y_batch = []
                    del_list = []
                    #인덱스 도달한거 삭제 및 새로운 이미지 데이터 추가
                    for i in range(len(idx_buffer)):
                        if idx_buffer[i] < last_idx_buffer[i]:
                            del x_buffer[i*self.SEQ_LENGTH]
                            x_buffer.insert(i*self.SEQ_LENGTH , self.get_data(idx_buffer[i],code_buffer[i]))
                            y_batch.append(y_buffer[i][np.where(y_buffer[i] == idx_buffer[i])[0][0], 1])
                            idx_buffer[i] += 1
                        else:
                            del_list.append(i)
                    if del_list:
                        del_list.reverse()
                        for i in del_list:
                            del x_buffer[i*self.SEQ_LENGTH: (i+1)*self.SEQ_LENGTH],y_buffer[i],idx_buffer[i],last_idx_buffer[i],code_buffer[i]
                    #새로 추가
                    while self.has_data and len(idx_buffer) < self.BATCH_SIZE:
                        self.append_chartdata_in_batch(list,x_buffer,y_buffer,idx_buffer,last_idx_buffer,code_buffer,y_batch)
                    if len(y_batch) <= 0:
                        break
                    #세션 실행
                    cost, _ = sess.run([self.cost, self.optimizer], feed_dict={
                        self.inImage: x_buffer, self.out: y_batch, self.in_batch_size: len(y_batch), self.phase : True })
                    cnt += 1
                    if cnt % 500 == 0 :
                        print('%s epoch %s times mean error : %s' % (epoch, cnt, cost))

                '''
                for code in list:

                    self.chart_code = code

                    try:
                        self.chart_data_y = np.genfromtxt("./chart_data_y/" + self.chart_code + ".csv", delimiter=',')
                    except OSError:
                        continue
                    last_idx = 180 + int( (self.chart_data_y.shape[0]) * 0.8 )
                    idx = 180 + TrendPredictNetwork.SEQ_LENGTH - 1
                    print('buffer load, code : %s, idx : %s, last_idx : %s' % (self.chart_code, idx, last_idx))
                    input_x = [ self.get_data(i) for i in range(180, idx)]
                    while idx < last_idx:
                        #print('%s %s ' % (self.chart_code,idx))
                        input_x.append(self.get_data(idx))
                        input_y = [self.chart_data_y[np.where(self.chart_data_y == idx)[0][0], 1]]
                        cost, _ = sess.run([self.cost, self.optimizer],
                                           feed_dict={self.inImage: input_x, self.out: input_y})
                        idx += 1
                        del input_x[0]
                    print()
               '''

                saver.save(sess, self.MODEL_PATH[self.data_type]+'/model_Predict.cptk')
            if epoch % 300 == 0:
                # 정확도 확인 함수
                self.validation(sess,epoch)
    def validation(self, sess, epoch):
        print('start %s epoch validation' % epoch)
        cnt = 0
        correct_cnt = 0
        list = deepcopy(self.chartcode_list)
        random.shuffle(list)
        list=list[:len(list)//self.DIVIVE_DATA]
        x_buffer = []
        y_buffer = []

        idx_buffer = []
        last_idx_buffer = []
        code_buffer = []
        self.has_data = True

        while True:
            y_batch = []
            del_list = []
            # 인덱스 도달한거 삭제 및 새로운 이미지 데이터 추가
            for i in range(len(idx_buffer)):
                if idx_buffer[i] < last_idx_buffer[i]:
                    del x_buffer[i * self.SEQ_LENGTH]
                    x_buffer.insert(i * self.SEQ_LENGTH, self.get_data(idx_buffer[i], code_buffer[i]))
                    y_batch.append(y_buffer[i][np.where(y_buffer[i] == idx_buffer[i])[0][0], 1])
                    idx_buffer[i] += 1
                else:
                    del_list.append(i)
            if del_list:
                del_list.reverse()
                for i in del_list:
                    del x_buffer[i * self.SEQ_LENGTH: (i + 1) * self.SEQ_LENGTH], y_buffer[i], idx_buffer[i], last_idx_buffer[i], code_buffer[i]
            # 새로 추가
            while self.has_data and len(idx_buffer) < self.BATCH_SIZE:
                self.append_chartdata_in_batch(list, x_buffer, y_buffer, idx_buffer, last_idx_buffer,
                                               code_buffer, y_batch, is_train=False)
            if len(y_batch) <= 0:
                break
            # 세션 실행
            p_y_list = sess.run(self.predict, feed_dict={self.inImage: x_buffer, self.in_batch_size: len(y_batch), self.phase : False})

            for i, p_y in enumerate(p_y_list):
                cnt += 1
                if (y_batch[i] == p_y):
                    correct_cnt += 1

            print('epoch %s 정확도 : %s (%s/%s)' % (epoch, ((correct_cnt / cnt) * 100), correct_cnt, cnt))

    def predict(self):
        pass
    def load_model(self):
        pass

    def get_data(self, idx, code):
        return self.get_image(code, idx, self.model_type)
    def get_image(self, code, idx, days=20):
        filepath = self.INPUT_IMAGE_PATH[self.data_type] + "%s_%s_%s.png" % (code, days, idx)
        try:
            with pilimg.open(filepath) as im_file:
                im = np.asarray(im_file)
                # 가로픽셀 추가
                xPixel = Environment.RANGE_SHAPES[days][self.data_type][1]
                yPixel = Environment.RANGE_SHAPES[days][self.data_type][0]
                if im.shape[1] < xPixel:
                    while True:
                        try:
                            xarray = np.array([[[255, 255, 255, 255]] * (xPixel - im.shape[1])] * im.shape[0])
                            im = np.hstack([im, xarray])
                            break
                        except:
                            print('y축 이미지 추가 실패', xarray.shape, im.shape)
                    del xarray
                elif im.shape[1] > xPixel:
                    im=im[:,:xPixel]
                # 세로픽셀 추가
                if im.shape[0] < yPixel:
                    while True:
                        try:
                            yarray = np.array([[[255, 255, 255, 255]] * (im.shape[1])] * (yPixel - im.shape[0]))
                            im = np.vstack([im, yarray])
                            break
                        except:
                            print('y축 이미지 추가 실패', yarray.shape, im.shape)
                    del yarray
                elif im.shape[0] > yPixel:
                    im = im[:yPixel]

            return im
        except:
            return None
    # 테스트 코드
    def test(self):
        #config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=1,reshape=True)

            print('Loading Model')
            ckpt = tf.train.get_checkpoint_state(self.MODEL_PATH[self.data_type])
            saver.restore(sess, ckpt.model_checkpoint_path)
            self.validation(sess,0)

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



class Actor:
    def __init__(self, action_size, learning_rate = 0.01, model_type = 1):
        #매개변수 : 입력값 크기(이미지), 출력개수, 학습률
        self.inImage = tf.placeholder(tf.float32,shape=[None]+Environment.RANGE_SHAPE[model_type])
        self.phase= tf.placeholder(tf.bool, name='phase')

        conv1 = tf.contrib.layers.convolution2d(inputs=self.inImage, num_outputs=32, kernel_size=[8, 8],
                                                padding="VALID", biases_initializer=None)
        mp1 = tf.contrib.layers.max_pool2d(conv1, kernel_size=[2, 2])
        conv2 = tf.contrib.layers.convolution2d(inputs=mp1, num_outputs=64, kernel_size=[4, 4],
                                                padding="VALID", biases_initializer=None)
        mp2 = tf.contrib.layers.max_pool2d(conv2, kernel_size=[2, 2])
        conv3 = tf.contrib.layers.convolution2d(inputs=mp2, num_outputs=64, kernel_size=[3, 3],
                                                padding="VALID", biases_initializer=None)
        mp3 = tf.contrib.layers.max_pool2d(conv3, kernel_size=[2, 2])
        conv4 = tf.contrib.layers.convolution2d(inputs=mp3, num_outputs=32, kernel_size=[7, 7],
                                                padding="VALID", biases_initializer=None)
        mp4 = tf.contrib.layers.max_pool2d(conv4, kernel_size=[2, 2])

        img_result = tf.reshape(mp4, [-1, mp4.get_shape().as_list()[0]])

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
        rnn_out = tf.reshape(lstm_outputs, [-1, 1024])

        batch_norm = tf.contrib.layers.batch_norm(rnn_out, is_training=self.phase, scope='bn', center=True,
                                                  scale=True)
        fc_layer_1 = slim.fully_connected(batch_norm, 256, activation_fn=tf.nn.leaky_relu,
                                          weights_initializer=normalized_columns_initializer(), biases_initializer=None)


        self.action = slim.fully_connected(fc_layer_1, action_size, activation_fn=None,
                                           weights_initializer=normalized_columns_initializer(.01),
                                           biases_initializer=None)

        #학습관련 신경망



class Critic:
    def __init__(self, action_size, learning_rate = 0.01, model_type = 1):
        #매개변수 : 입력값 크기(이미지), 출력개수, 학습률
        self.inImage = tf.placeholder(tf.float32,shape=[None]+Environment.RANGE_SHAPE[model_type])
        self.phase= tf.placeholder(tf.bool, name='phase')

        conv1 = tf.contrib.layers.convolution2d(inputs=self.inImage, num_outputs=32, kernel_size=[8, 8],
                                                padding="VALID", biases_initializer=None)
        mp1 = tf.contrib.layers.max_pool2d(conv1, kernel_size=[2, 2])
        conv2 = tf.contrib.layers.convolution2d(inputs=mp1, num_outputs=64, kernel_size=[4, 4],
                                                padding="VALID", biases_initializer=None)
        mp2 = tf.contrib.layers.max_pool2d(conv2, kernel_size=[2, 2])
        conv3 = tf.contrib.layers.convolution2d(inputs=mp2, num_outputs=64, kernel_size=[3, 3],
                                                padding="VALID", biases_initializer=None)
        mp3 = tf.contrib.layers.max_pool2d(conv3, kernel_size=[2, 2])
        conv4 = tf.contrib.layers.convolution2d(inputs=mp3, num_outputs=32, kernel_size=[7, 7],
                                                padding="VALID", biases_initializer=None)
        mp4 = tf.contrib.layers.max_pool2d(conv4, kernel_size=[2, 2])

        img_result = tf.reshape(mp4, [-1, mp4.get_shape().as_list()[0]])

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
        rnn_out = tf.reshape(lstm_outputs, [-1, 1024])

        batch_norm = tf.contrib.layers.batch_norm(rnn_out, is_training=self.phase, scope='bn', center=True,
                                                  scale=True)
        fc_layer_1 = slim.fully_connected(batch_norm, 256, activation_fn=tf.nn.leaky_relu,
                                          weights_initializer=normalized_columns_initializer(), biases_initializer=None)


        self.value = slim.fully_connected(fc_layer_1, 1, activation_fn=None,
                                           weights_initializer=normalized_columns_initializer(1.0),
                                           biases_initializer=None)

