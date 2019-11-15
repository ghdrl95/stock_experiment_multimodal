'''
정책별 Q값을 모아서 풀커넥트 레이어를 지나 Q값을 연산하는 방식
'''
from network_4 import updateTargetGraph, experience_buffer, updateTarget,TrendPredictNetwork
import os
import tensorflow as tf
from agent import Agent
import numpy as np
from environment import Environment
import sys
from time import sleep
import json
class PolicyLearner:
    replay_memory = 20
    def __init__(self,load_model=True, learning_rate = 0.01, min_trading_unit=0, max_trading_unit=300, delayed_reward_threshold=.01, training = True):
        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit
        self.delayed_reward_threshold = delayed_reward_threshold


        self.batch_size = 2
        self.update_freq = 4
        self.y = .99
        self.discount_factor= .8 #0.8**30 = 0.004
        self.startE = 1
        self.endE = 0.1
        self.anneling_steps = 2000.
        self.num_episodes = 20000 // 40
        self.pre_train_steps = 2000
        self.max_epLength = 20

        self.training_step= 5

        self.pre_buffer_size = self.max_epLength * self.pre_train_steps

        self.load_model = load_model
        self.path = './dqn_basic_sto_renew_Q_4'

        # 모델을 세이브할 장소를 만든다.
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # self.h_size = 512
        self.tau = 0.001

        tf.reset_default_graph()

        self.total_steps = 0
        self.network_type = 20
        self.data_type = [Environment.TYPE_BASIC, Environment.TYPE_STO, Environment.TYPE_DMI]

        self.buffer_size = 0
        for image_type in self.data_type:
            image_size = 1
            for shape in Environment.RANGE_SHAPES[self.network_type][image_type]:
                image_size *= shape
            self.buffer_size+=image_size

        self.buffer_size = ((25*(1024**3)) // (self.buffer_size*2))//10 * 10 #10GB / Imagesize
        print(self.buffer_size)
        self.mainQN = TrendPredictNetwork(learning_rate=learning_rate, model_type=self.network_type, name='main_%s_'%(self.network_type),data_type=self.data_type)
        if training:
            self.targetQN = TrendPredictNetwork(learning_rate=learning_rate, model_type=self.network_type, name='target_%s_'%(self.network_type),data_type=self.data_type)


    def train(self,sess,number):
        environment = Environment()
        agent = Agent(environment,
                           min_trading_unit=self.min_trading_unit,
                           max_trading_unit=self.max_trading_unit,
                           delayed_reward_threshold=self.delayed_reward_threshold)
        trainables = tf.trainable_variables()
        targetOps = updateTargetGraph(trainables, self.tau)
        rList = []
        #portfolio_list=[]

        episode_buffer = experience_buffer()

        stepDrop = (self.startE - self.endE) / self.anneling_steps

        if self.load_model == True:
            e = self.endE
        else:
            e = self.startE

        # 주요 신경망과 동일하게 타겟 신경망을 설정한다
        updateTarget(targetOps, sess)
        # 에피소드 시작
        for ii in range(self.num_episodes):
            rAll = 0
            d = False
            j = 0
            episode_buffer.buffer = []
            episode_reward_buffer = []
            environment.reset()
            agent.reset()
            #print('%d 번째 episode 초기화 :' % ii,self.environment.idx, self.environment.KOSPI_idx, 'total num :',total_steps, '종목코드',self.environment.chart_code)
            s = [environment.get_image(self.network_type,datatype) for datatype in self.data_type]
            s_potfol = np.array(agent.get_states())

            episode_step = 1

            while j < self.max_epLength and not d:

                j += 1

                #입력값으로 행동선택하기(베이시안 + 볼트만)
                #all_Q_d = np.zeros([self.agent.NUM_ACTIONS])
                feed_dict = { self.mainQN.temp:e, self.mainQN.keep_per:(1-e)+0.1, self.mainQN.phase:True}
                for i, _ in enumerate(self.data_type):
                    feed_dict[self.mainQN.inImage[i]] = [s[i]]
                Q_d = sess.run(self.mainQN.Q_dist, feed_dict=feed_dict)
                '''
                for i, mainQN in enumerate(self.mainQN):
                    Q_d = sess.run(mainQN.Q_dist, feed_dict={mainQN.inImage:[s[i]],
                                                                                              mainQN.portfolio_state:[s_potfol],
                                                                                              mainQN.temp:e,
                                                                                              mainQN.keep_per:(1-e)+0.1,
                                                                                             mainQN.phase:True})
                    all_Q_d += Q_d[0]
                #모든 신경망의 확률값을 더한 뒤 나눔
                #print(np.sum(all_Q_d))
                all_Q_d /= len(self.data_type)
                all_Q_d /= np.sum(all_Q_d)'''
                #print(np.sum(all_Q_d))
                Q_d = Q_d[0]
                a = np.random.choice(Q_d, p=Q_d)
                action = np.argmax(Q_d == a)
                #정책에 행동전달
                delayed_reward = agent.act(action=action, confidence=Q_d[action])
                d = environment.step()
                if e > self.endE and self.total_steps > self.pre_train_steps:
                    e -= stepDrop
                '''
                immediate_reward, delayed_reward = self.agent.act(action=action, confidence=all_Q_d[action])

                if e > self.endE and total_steps > self.pre_train_steps:
                    e -= stepDrop


                #다음 인덱스로 넘어가기
                d = self.environment.step()
                if (delayed_reward == 0 and episode_step % 5 == 0) or d:
                    delayed_reward = immediate_reward
                    self.agent.base_portfolio_value = self.agent.portfolio_value
              '''
                #다음이미지,포폴 받기
                #print('total step :', total_steps, 'current episode step : ', j, 'idx :', self.environment.idx, 'kospi_idx', self.environment.KOSPI_idx, '종목코드',self.environment.chart_code)
                s1 = [environment.get_image(self.network_type,datatype) for datatype in self.data_type]
                s1_potfol = np.array(agent.get_states())

                episode_reward_buffer.append(0)
                #버퍼에 저장
                # 원래 버퍼 순서: 상태 액션 보상 다음상태 종료여부
                # 수정 버퍼    : 현재이미지 액션 보상 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴
                # 재수정 버퍼    : 현재이미지 액션 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴 보상(디스카운트 펙터설정)
                #episode_buffer.add([s, action, delayed_reward, s1, s1_potfol, d, before_rnn_state, rnn_state, s_potfol  ]  )
                episode_buffer.buffer.append([s, action, s1, s1_potfol, d, s_potfol])
                if len(self.myBuffer.buffer) > self.pre_buffer_size and j % self.training_step == 0:
                    try:
                        #버퍼에서 데이터 가져오기
                        # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신

                        #원래 버퍼 순서: 상태 액션 보상 다음상태 종료여부
                        # 수정 버퍼    : 현재이미지 액션 보상 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴
                        # 배치 학습 데이터 크기
                        trainBatch, size = self.myBuffer.sample(self.replay_memory)#,rList)#(self.batch_size)
                        #print('훈련데이터 추출 결과 : ', trainBatch.shape)
                        #보상을 전행동에 영향이 가도록 할인인자로 곱해야함



                        # 아래는 target Q-value를 업데이트하는 Double-DQN을 수행한다
                        # 주요 신경망에서 행동을 고른다.
                        #학습 시 베이시안과 볼트만을 사용하지 않음

                        #LSTM 학습을 위해서 랜덤한 에피소드에 랜덤한 날짜부터 replay memory만큼 선정하고 사용함
                        # 재수정 버퍼    : 현재이미지 액션 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴 보상(디스카운트 펙터설정)

                        feed_dict = {#self.mainQN.portfolio_state: [data for data in trainBatch[:,3]],
                                                 self.mainQN.keep_per: 1.0,
                                                 self.mainQN.phase: True}
                        feed_dict_2 = {#self.targetQN.portfolio_state: [data for data in trainBatch[:, 3]],
                                       self.targetQN.keep_per: 1.0,
                                       self.targetQN.phase: True}
                        for i in range(len(self.data_type)):
                            feed_dict[self.mainQN.inImage[i]] = [datas[i] for datas in trainBatch[:,2]]
                            feed_dict_2[self.targetQN.inImage[i]]= [datas[i] for datas in trainBatch[:, 2]]
                        Q1 = sess.run(self.mainQN.predict,
                                      feed_dict=feed_dict)

                        Q2 = sess.run(self.targetQN.Qout,  # feed_dict 수정해야함
                                      feed_dict=feed_dict_2)
                        del feed_dict_2,feed_dict

                        # 종료 여부에 따라 가짜 라벨을 만들어준다
                        end_multiplier = -(trainBatch[:, 4] - 1)
                        # 타겟 신경망의 Q 값들 중에 주요 신경망에서 고른 행동 번째의 Q 값들을 가져온다.(이부분이 doubleQ)
                        doubleQ = Q2[range(size), Q1]
                        # 보상에 대한 더블 Q 값을 더해준다. y는 할인 인자
                        # targetQ 는 즉각적인 보상 + 다음 상태의 최대 보상(doubleQ)
                        targetQ = trainBatch[:, 6] + (self.y * doubleQ * end_multiplier)
                        # 우리의 타겟 값들과 함께 신경망을 업데이트해준다.
                        # 행동들에 대해서 targetQ 값과의 차이를 통해 손실을 구하고 업데이트
                        # 원래 버퍼 순서: 상태 액션 보상 다음상태 종료여부
                        # 수정 버퍼    : 현재이미지 액션 보상 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴

                        feed_dict3 = {#self.mainQN.portfolio_state: [data for data in trainBatch[:, 5]],
                                      self.mainQN.targetQ: targetQ,
                                      self.mainQN.actions: trainBatch[:, 1],
                                      self.mainQN.keep_per: 1.0,
                                      self.mainQN.phase: True}
                        for i in range(len(self.data_type)):
                            feed_dict3[self.mainQN.inImage[i]] =  [datas[i] for datas in trainBatch[:, 0]]
                        _ = sess.run(self.mainQN.updateModel, feed_dict=feed_dict3)
                        del feed_dict3
                        '''
                            _ = sess.run(self.mainQN[i].updateModel, \
                                         feed_dict={self.mainQN[i].inImage: np.vstack(trainBatch[:, 0]),
                                                    self.mainQN[i].targetQ: targetQ,
                                                    self.mainQN[i].actions: trainBatch[:, 1]})
                        '''
                        del trainBatch, size

                    except IndexError as eee:
                        print(eee)


                rAll += delayed_reward

                #rAll = delayed_reward
                # 상태를 바꾼다.
                del s
                s = s1
                del s_potfol
                s_potfol = s1_potfol
                self.total_steps += 1
                episode_step += 1



            #할인인자 적용한 보상을 에피소드 버퍼에 추가
            #if delayed_reward != 0:
            accumulate = 0
            if delayed_reward == 0 and environment.chart_y_cnt>10:
                delayed_reward = (- 0.1) * environment.chart_y_cnt

            episode_reward_buffer[-1] = delayed_reward
            episode_reward_buffer.reverse()
            #print('%s episode_reward_len : ' % ii, len(episode_reward_buffer), 'episode_buffer_len :', len(episode_buffer.buffer))
            for i, reward in enumerate(episode_reward_buffer):
                accumulate = self.discount_factor * accumulate + reward
                idx = -(i+1)
                episode_buffer.buffer[idx] += [accumulate]
                #print(idx, len(episode_buffer.buffer[idx]))

            self.myBuffer.add(episode_buffer.buffer,type=1)

            if len(rList)+1>= self.buffer_size:
                # self.buffer[0:1] = []
                del rList[0]
            rList.append(abs(delayed_reward))
            #environment.chartcode_value[environment.chart_code] += 1 if agent.portfolio_value > agent.initial_balance else -1
            print("%d %d %s %s %d %d %d %d %d" % (ii, environment.chart_y_cnt, environment.chart_code, delayed_reward, agent.portfolio_value, agent.minimum_portfolio_value, agent.maximum_portfolio_value, environment.idx-self.max_epLength, environment.idx))
            #print("%d %4f %d %4f %4f %d %d"% (total_steps, np.mean(rList[-10:]), np.mean(portfolio_list), np.max(rList[-10:]),np.min(rList[-10:]),np.max(portfolio_list),np.min(portfolio_list)))#e)
                #print(sys.getsizeof(myBuffer.buffer), sys.getsizeof(episode_buffer.buffer))
            #portfolio_list= []
            if self.total_steps > self.pre_train_steps and ii % 250 == 0 and number == 0:
                try:
                    self.saver.save(sess, self.path + '/model-' + str(ii) + '.cptk')
                    with open('./value_chart.txt','w') as f:
                        data = json.dumps(environment.chartcode_value)
                        f.write(data)
                    del data
                    #print("Saved Model")
                except:
                    pass
            updateTarget(targetOps, sess)
        # 학습 끝 평균 보상을 표시
        if number == 0:
            self.saver.save(sess, self.path + '/model-' + str(ii) + '.cptk')
        print("평균 episode 별 보상 값 : " + str(sum(rList) / self.num_episodes) )
    def test(self, sess):
        environment = Environment()
        agent = Agent(environment,
                      min_trading_unit=self.min_trading_unit,
                      max_trading_unit=self.max_trading_unit,
                      delayed_reward_threshold=self.delayed_reward_threshold)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1,reshape=True)


        #portfolio_list=[]
        total_steps = 0

        e = self.endE


        # 변수를 초기화한다.
        sess.run(init)
        print('Loading Model...')
        # 모델을 불러온다
        ckpt = tf.train.get_checkpoint_state(self.path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        # 에피소드 시작
        for ii, code in enumerate(environment.chartcode_list):
            rAll = 0
            d = False
            j = 0

            #print('%d번째 %s episode' % (ii, code))
            if not environment.reset(code):
                continue
            agent.reset()

            s = [environment.get_image(self.network_type,datatype) for datatype in self.data_type]
            s_potfol = np.array(agent.get_states())

            episode_step = 1

            while j < self.max_epLength and not d:

                j += 1

                #입력값으로 행동선택하기(베이시안 + 볼트만)
                #all_Q_d = np.zeros([self.agent.NUM_ACTIONS])
                feed_dict = {#self.mainQN.portfolio_state:[s_potfol],
                             self.mainQN.temp:e, self.mainQN.keep_per:(1-e)+0.1, self.mainQN.phase:False}
                for i, _ in enumerate(self.data_type):
                    feed_dict[self.mainQN.inImage[i]] = [s[i]]
                Q_d = sess.run(self.mainQN.Q_dist, feed_dict=feed_dict)
                '''
                for i, mainQN in enumerate(self.mainQN):
                    Q_d = sess.run(mainQN.Q_dist, feed_dict={mainQN.inImage:[s[i]],
                                                                                              mainQN.portfolio_state:[s_potfol],
                                                                                              mainQN.temp:e,
                                                                                              mainQN.keep_per:(1-e)+0.1,
                                                                                             mainQN.phase:True})
                    all_Q_d += Q_d[0]
                #모든 신경망의 확률값을 더한 뒤 나눔
                #print(np.sum(all_Q_d))
                all_Q_d /= len(self.data_type)
                all_Q_d /= np.sum(all_Q_d)'''
                #print(np.sum(all_Q_d))
                Q_d = Q_d[0]
                a = np.random.choice(Q_d, p=Q_d)
                #a = max(Q_d)
                action = np.argmax(Q_d == a)
                #정책에 행동전달
                delayed_reward = agent.act(action=action, confidence=Q_d[action])
                d = environment.step()

                s1 = [environment.get_image(self.network_type,datatype) for datatype in self.data_type]
                s1_potfol = np.array(agent.get_states())

                rAll += delayed_reward

                #rAll = delayed_reward
                # 상태를 바꾼다.
                del s
                s = s1
                del s_potfol
                s_potfol = s1_potfol
                total_steps += 1
                episode_step += 1
            print("%d %d %s %s %d %d %d" % (ii, environment.chart_y_cnt, environment.chart_code, delayed_reward, agent.portfolio_value, agent.minimum_portfolio_value, agent.maximum_portfolio_value))
    def test_one(self, sess):
        environment = Environment()
        agent = Agent(environment,
                      min_trading_unit=self.min_trading_unit,
                      max_trading_unit=self.max_trading_unit,
                      delayed_reward_threshold=self.delayed_reward_threshold)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1,reshape=True)


        #portfolio_list=[]
        total_steps = 0

        e = self.endE


        # 변수를 초기화한다.
        sess.run(init)
        print('Loading Model...')
        # 모델을 불러온다
        ckpt = tf.train.get_checkpoint_state(self.path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        # 에피소드 시작
        for ii, code in enumerate(environment.chartcode_list):
            rAll = 0
            d = False
            j = 0

            #print('%d번째 %s episode' % (ii, code))
            if not environment.reset(code):
                continue
            agent.reset()

            s = [environment.get_image(self.network_type,datatype) for datatype in self.data_type]
            s_potfol = np.array(agent.get_states())

            episode_step = 1

            while j < self.max_epLength and not d:

                j += 1

                #입력값으로 행동선택하기(베이시안 + 볼트만)
                #all_Q_d = np.zeros([self.agent.NUM_ACTIONS])
                feed_dict = {#self.mainQN.portfolio_state:[s_potfol],
                             self.mainQN.temp:e, self.mainQN.keep_per:(1-e)+0.1, self.mainQN.phase:False}
                for i, _ in enumerate(self.data_type):
                    feed_dict[self.mainQN.inImage[i]] = [s[i]]
                Q_d = sess.run(self.mainQN.Q_dist, feed_dict=feed_dict)
                '''
                for i, mainQN in enumerate(self.mainQN):
                    Q_d = sess.run(mainQN.Q_dist, feed_dict={mainQN.inImage:[s[i]],
                                                                                              mainQN.portfolio_state:[s_potfol],
                                                                                              mainQN.temp:e,
                                                                                              mainQN.keep_per:(1-e)+0.1,
                                                                                             mainQN.phase:True})
                    all_Q_d += Q_d[0]
                #모든 신경망의 확률값을 더한 뒤 나눔
                #print(np.sum(all_Q_d))
                all_Q_d /= len(self.data_type)
                all_Q_d /= np.sum(all_Q_d)'''
                #print(np.sum(all_Q_d))
                Q_d = Q_d[0]
                a = np.random.choice(Q_d, p=Q_d)
                action = np.argmax(Q_d == a)
                #정책에 행동전달
                delayed_reward = agent.act(action=action, confidence=Q_d[action])
                d = environment.step()

                s1 = [environment.get_image(self.network_type,datatype) for datatype in self.data_type]
                s1_potfol = np.array(agent.get_states())

                rAll += delayed_reward

                #rAll = delayed_reward
                # 상태를 바꾼다.
                del s
                s = s1
                del s_potfol
                s_potfol = s1_potfol
                total_steps += 1
                episode_step += 1
            print("%d %d %s %s %d %d %d" % (ii, environment.chart_y_cnt, environment.chart_code, delayed_reward, agent.portfolio_value, agent.minimum_portfolio_value, agent.maximum_portfolio_value))
import threading
if __name__ == "__main__":
    obj = PolicyLearner(load_model=False)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        obj.saver = tf.train.Saver(max_to_keep=1, reshape=True)

        obj.myBuffer = experience_buffer(obj.buffer_size)

        sess.run(init)
        '''
        if obj.load_model == True:
            print('Loading Model...')
            # 모델을 불러온다
            ckpt = tf.train.get_checkpoint_state(obj.path)
            obj.saver.restore(sess, ckpt.model_checkpoint_path)

        threads = [threading.Thread(target=obj.train,args=[sess,_]) for _ in range(40)]
        print("학습시작")
        for thread in threads:
            thread.start()
        print("학습완료 대기")
        for thread in threads:
            thread.join()
        #obj.train(sess)
        #'''
        print("학습완료 및 결과보기")
        obj.test(sess)

















