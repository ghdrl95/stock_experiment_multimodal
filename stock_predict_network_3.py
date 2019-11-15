'''
이미지를 열단위로 짤라서 LSTM에 넣는 방식
'''
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from copy import deepcopy
import random
import PIL.Image as pilimg
import os
import json

class TrendPredictNetwork:
    chartcode_list = ["000060","000080","000087","000227","000240","000325","000500","000520","000547","000760","000860","000880","000885","000950","000995","001060","001065","001080","001120","001140","001260","001270","001275","001390","001430","001529","001550","001680","001770","001790","001799","001880","002020","002025","002100","002200","002210","002220","002240","002250","002300","002390","002460","002620","002795","002810","002995","003000","003075","003090","003220","003350","003490","003495","003960","004360","004380","004430","004545","004700","004960","004970","005070","005190","005257","005420","005820","005850","005880","005940","006120","006125","007340","007540","007570","007575","008355","008490","008500","009140","009420","009470","009770","009830","010660","010780","011155","011760","011785","011790","011810","012450","012630","013580","014280","015760","016360","016385","016580","016800","019440","020150","023450","023590","024110","024720","026960","027390","028100","029460","029780","030000","030200","032640","033240","033270","034120","034220","034300","034310","034590","035250","042660","044820","047050","047810","051600","052690","055490","058430","063160","064350","066575","068290","069260","069500","069660","071090"]#,"078000","078935","079430","079440","081000","084670","091160","091180","091230","093050","099140","102110","102460","103140","105190","105630","108590","108675","111770","112610","114090","115390","117690","120115","122090","131890","138540","139260","139290","140950","143850","143860","145850","145995","147970","148020","152100","152330","152870","153270","159800","168300","168580","169950","174360","180640","183710","185680","192720","194370","195870","200020","200030","200040","203780","210540","210980","211260","211560","213500","217790","226490","227540","227830","227840","234080","237350","241560","243880","245340","248170","249420","260200","261220","261920","265690","266420","267290","272450","272550","275540","277630","281820","293180","294400","302450","305050","306520","310960","315270","316140","500009","500034","510008","520004","520009","520010","520025","530014","550009","550014","550051","590007","590009"]
    chartcode_value = {"000060": 1,"000080": 1,"000087": 1,"000227": 1,"000240": 1,"000325": 1,"000500": 1,"000520": 1,"000547": 1,"000760": 1,"000860": 1,"000880": 1,"000885": 1,"000950": 1,"000995": 1,"001060": 1,"001065": 1,"001080": 1,"001120": 1,"001140": 1,"001260": 1,"001270": 1,"001275": 1,"001390": 1,"001430": 1,"001529": 1,"001550": 1,"001680": 1,"001770": 1,"001790": 1,"001799": 1,"001880": 1,"002020": 1,"002025": 1,"002100": 1,"002200": 1,"002210": 1,"002220": 1,"002240": 1,"002250": 1,"002300": 1,"002390": 1,"002460": 1,"002620": 1,"002795": 1,"002810": 1,"002995": 1,"003000": 1,"003075": 1,"003090": 1,"003220": 1,"003350": 1,"003490": 1,"003495": 1,"003960": 1,"004360": 1,"004380": 1,"004430": 1,"004545": 1,"004700": 1,"004960": 1,"004970": 1,"005070": 1,"005190": 1,"005257": 1,"005420": 1,"005820": 1,"005850": 1,"005880": 1,"005940": 1,"006120": 1,"006125": 1,"007340": 1,"007540": 1,"007570": 1,"007575": 1,"008355": 1,"008490": 1,"008500": 1,"009140": 1,"009420": 1,"009470": 1,"009770": 1,"009830": 1,"010660": 1,"010780": 1,"011155": 1,"011760": 1,"011785": 1,"011790": 1,"011810": 1,"012450": 1,"012630": 1,"013580": 1,"014280": 1,"015760": 1,"016360": 1,"016385": 1,"016580": 1,"016800": 1,"019440": 1,"020150": 1,"023450": 1,"023590": 1,"024110": 1,"024720": 1,"026960": 1,"027390": 1,"028100": 1,"029460": 1,"029780": 1,"030000": 1,"030200": 1,"032640": 1,"033240": 1,"033270": 1,"034120": 1,"034220": 1,"034300": 1,"034310": 1,"034590": 1,"035250": 1,"042660": 1,"044820": 1,"047050": 1,"047810": 1,"051600": 1,"052690": 1,"055490": 1,"058430": 1,"063160": 1,"064350": 1,"066575": 1,"068290": 1,"069260": 1,"069500": 1,"069660": 1,"071090": 1,"078000": 1,"078935": 1,"079430": 1,"079440": 1,"081000": 1,"084670": 1,"091160": 1,"091180": 1,"091230": 1,"093050": 1,"099140": 1,"102110": 1,"102460": 1,"103140": 1,"105190": 1,"105630": 1,"108590": 1,"108675": 1,"111770": 1,"112610": 1,"114090": 1,"115390": 1,"117690": 1,"120115": 1,"122090": 1,"131890": 1,"138540": 1,"139260": 1,"139290": 1,"140950": 1,"143850": 1,"143860": 1,"145850": 1,"145995": 1,"147970": 1,"148020": 1,"152100": 1,"152330": 1,"152870": 1,"153270": 1,"159800": 1,"168300": 1,"168580": 1,"169950": 1,"174360": 1,"180640": 1,"183710": 1,"185680": 1,"192720": 1,"194370": 1,"195870": 1,"200020": 1,"200030": 1,"200040": 1,"203780": 1,"210540": 1,"210980": 1,"211260": 1,"211560": 1,"213500": 1,"217790": 1,"226490": 1,"227540": 1,"227830": 1,"227840": 1,"234080": 1,"237350": 1,"241560": 1,"243880": 1,"245340": 1,"248170": 1,"249420": 1,"260200": 1,"261220": 1,"261920": 1,"265690": 1,"266420": 1,"267290": 1,"272450": 1,"272550": 1,"275540": 1,"277630": 1,"281820": 1,"293180": 1,"294400": 1,"302450": 1,"305050": 1,"306520": 1,"310960": 1,"315270": 1,"316140": 1,"500009": 1,"500034": 1,"510008": 1,"520004": 1,"520009": 1,"520010": 1,"520025": 1,"530014": 1,"550009": 1,"550014": 1,"550051": 1,"590007": 1,"590009": 1  }
    # 하이퍼파라미터 설정
    EPOCH_NUM = 100
    RANGE_SHAPE = {5 : [780, 130, 4], 20 : [630, 245, 4], 60 : [780, 550, 4], 6 : [780, 130, 4], 7 : [780, 130, 4], 25 : [780, 280, 4]}
    MODEL_PATH = './stock_predict_test3'
    SEQ_LENGTH = 1
    INPUT_IMAGE_PATH = 'C:/chart_images_other/'
    OUTPUT_DATA_PATH = "./chart_data_y/"
    BATCH_SIZE = 50
    LEARNING_RATE = 0.001
    LSTM_LAYERS = 5
    DEVICES=['/cpu:0','/gpu:0']
    def __init__(self, model_type, is_train = True):
        with open('./value_chart.txt', 'r') as f:
            lines = f.readlines()
            data = ''
            for line in lines:
                data += line
            self.chartcode_value = json.loads(data)

        # 신경망 구성
        self.model_type = model_type
        device = self.DEVICES[1] if is_train else self.DEVICES[0]
        #self.SEQ_LENGTH = self.SEQ_LENGTH if is_train else 1
        with tf.device(device):
            with tf.variable_scope('predict_%s' % model_type):
                # 매개변수 : 입력값 크기(이미지), 출력개수, 학습률
                self.inImage = tf.placeholder(tf.float32, shape=[None] + TrendPredictNetwork.RANGE_SHAPE[model_type])
                self.out     = tf.placeholder(tf.int32, shape = [None])
                self.phase = tf.placeholder(tf.bool, name='phase')
                self.in_batch_size = tf.placeholder(tf.int32)
                out_onehot = tf.one_hot(self.out, 3) #[오른다, 그대로다, 내린다)


                # img_result = tf.reshape(mp4, [-1, mp4.get_shape().as_list()[0]])
                #img_result = tf.layers.flatten(batch_norm_5)
                img_result = tf.reshape( tf.transpose(self.inImage,perm=[0,2,1,3]),shape=[-1,self.inImage.shape[2],self.inImage.shape[1]*self.inImage.shape[3]])

                # 하나의 LSTML셀에 몇개의 뉴럴을 생성할 것인지 설정
                lstms=[]
                for i in range(self.LSTM_LAYERS):
                    lstm_cell = tf.contrib.rnn.LSTMCell(1024, state_is_tuple=True,trainable=is_train,name="LSTM_%d"%i)
                    lstms.append(lstm_cell)
                cell = tf.contrib.rnn.MultiRNNCell(lstms)

                lstm_outputs, lstm_state = tf.nn.dynamic_rnn(cell, img_result, dtype=tf.float32)

                #rnn_out = tf.reshape(lstm_outputs, [-1, 1024])[-1:]
                rnn_out = lstm_outputs[:,-1]
                output = slim.fully_connected(rnn_out, 3, activation_fn=None, trainable=is_train)
                hypothesis = tf.nn.softmax(output)
                self.predict = tf.argmax(hypothesis,axis=1)


                soft_cross = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=out_onehot)
                self.cost = tf.reduce_mean(soft_cross)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    #self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.cost)
                    self.optimizer = tf.contrib.opt.NadamOptimizer(self.LEARNING_RATE).minimize(self.cost)
    def append_chartdata_in_batch(self, list, x_buffer, y_buffer, idx_buffer, last_idx_buffer, code_buffer, y_list, is_train = True):
        if list:
            while True:
                chart_code = list[0]
                del list[0]
                if self.chartcode_value[chart_code] >= 0:
                    try:
                        chart_data_y = np.genfromtxt(self.OUTPUT_DATA_PATH + chart_code + ".csv", delimiter=',')
                    except OSError:
                        continue
                    break
                else:
                    self.chartcode_list.remove(chart_code)
            if is_train:
                last_idx = 180 + int((chart_data_y.shape[0]) * 0.8)
                idx = 180 + TrendPredictNetwork.SEQ_LENGTH - 1
            else:
                idx = 180 + int((chart_data_y.shape[0]) * 0.8)
                last_idx = 179 + chart_data_y.shape[0]
            print('buffer load, code : %s, idx : %s, last_idx : %s reamining data : %s' % (chart_code, idx, last_idx,len(list)))
            if is_train:
                input_x = [self.get_data(i, chart_code) for i in range(180, idx+1)]
            else:
                input_x = [self.get_data(i, chart_code) for i in range(idx-self.SEQ_LENGTH+1, idx+1)]

            x_buffer.extend(input_x)
            y_buffer.append(chart_data_y)
            idx_buffer.append(idx)
            last_idx_buffer.append(last_idx)
            code_buffer.append(chart_code)
            y_list.append(chart_data_y[np.where(chart_data_y == idx)[0][0], 1])
        else:
            self.has_data = False

    def train(self):

        #config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=1,reshape=True)
            for epoch in range(1, TrendPredictNetwork.EPOCH_NUM + 1):
                print('EPOCH %s start' % epoch)
                list = deepcopy(self.chartcode_list)
                random.shuffle(list)
                list= list[:int(len(list)/5)]
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

                saver.save(sess, self.MODEL_PATH+'/model_Predict.cptk')
            # 정확도 확인 함수
            self.validation(sess,epoch)
    def validation(self, sess, epoch):
        print('start %s epoch validation' % epoch)
        cnt = 0
        correct_cnt = 0
        list = deepcopy(self.chartcode_list)
        random.shuffle(list)
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
        filepath = self.INPUT_IMAGE_PATH+"%s_%s_%s.png" % (code, days, idx)
        with pilimg.open(filepath) as im_file:
            im = np.asarray(im_file)
            # 가로픽셀 추가
            xPixel = self.RANGE_SHAPE[days][1]
            yPixel = self.RANGE_SHAPE[days][0]
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

    # 테스트 코드
    def test(self):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            for epoch in range(TrendPredictNetwork.EPOCH_NUM):
                print('EPOCH %s start' % epoch)
                list = deepcopy(self.chartcode_list)
                random.shuffle(list)
                x_buffer = []
                y_buffer= []

                idx_buffer= []
                last_idx_buffer= []
                code_buffer= []
                self.has_data = True
                cnt = 0
                while True:
                    y_batch = []
                    #인덱스 도달한거 삭제 및 새로운 이미지 데이터 추가
                    for i in range(len(idx_buffer)):
                        if idx_buffer[i] < last_idx_buffer[i]:
                            del x_buffer[i*self.SEQ_LENGTH]
                            x_buffer.insert(i*self.SEQ_LENGTH , self.get_data(idx_buffer[i],code_buffer[i]))
                            y_batch.append(y_buffer[i][np.where(y_buffer[i] == idx_buffer[i])[0][0], 1])
                            idx_buffer[i] += 1
                        else:
                            del x_buffer[i],y_buffer[i],idx_buffer[i],last_idx_buffer[i],code_buffer[i]
                        #인덱스 추가
                    #새로 추가
                    while self.has_data and len(idx_buffer) < self.BATCH_SIZE:
                        self.append_chartdata_in_batch(list,x_buffer,y_buffer,idx_buffer,last_idx_buffer,code_buffer,y_batch)

                    #세션 실행
                    cost, _ = sess.run([self.cost, self.optimizer], feed_dict={
                        self.inImage: x_buffer, self.out: y_batch })
                    cnt += 1
                    if cnt % 100 == 0 :
                        print('mean error : %s' % (cost))
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
                if epoch % 200 == 0 :
                    self.validation(sess, epoch)
                    print('epoch : %s mean cost %s accuracy %s')
            saver.save(sess, self.MODEL_PATH+'/model_Predict.cptk')



obj = TrendPredictNetwork(20)
obj.train()




















