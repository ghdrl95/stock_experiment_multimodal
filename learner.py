from network import Qnetwork, updateTargetGraph, experience_buffer, updateTarget
import os
import tensorflow as tf
from agent import Agent
import numpy as np
from environment import Environment
import sys
from time import sleep
import json
class PolicyLearner:
    def __init__(self,load_model=True, learning_rate = 0.005, min_trading_unit=0, max_trading_unit=10, delayed_reward_threshold=.01, training = True):

        self.environment = Environment()
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)

        self.batch_size = 2
        self.update_freq = 4
        self.y = .99
        self.discount_factor= .8 #0.8**30 = 0.004
        self.startE = 1
        self.endE = 0.1
        self.anneling_steps = 10000.
        self.num_episodes = 10000
        self.pre_train_steps = 200
        self.max_epLength = 20
        self.replay_memory = 10
        self.training_step= 5

        self.load_model = load_model
        self.path = './dqn'

        # 모델을 세이브할 장소를 만든다.
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # self.h_size = 512
        self.tau = 0.001

        tf.reset_default_graph()

        self.network_type = [20,25]#, 6, 7]

        self.buffer_size = 0
        for image_type in self.network_type:
            image_size = 1
            for shape in self.environment.RANGE_SHAPE[image_type]:
                image_size *= shape
            self.buffer_size+=image_size

        self.buffer_size = ((15*(1024**3)) // (self.buffer_size*2*self.max_epLength))//10 * 10 #10GB / Imagesize
        print(self.buffer_size)
        self.mainQN = [Qnetwork(learning_rate=learning_rate, model_type=type, name='main_'+str(type)) for type in self.network_type]
        if training:
            self.targetQN = [Qnetwork(learning_rate=learning_rate, model_type=type, name='target_'+str(type)) for type in self.network_type]

        '''
        self.mainQN = [Qnetwork(learning_rate=learning_rate, model_type=5), Qnetwork(learning_rate=learning_rate, model_type=20),
                       Qnetwork(learning_rate=learning_rate, model_type=60)]
        self.targetQN = [Qnetwork(learning_rate=learning_rate, model_type=5), Qnetwork(learning_rate=learning_rate, model_type=20),
                         Qnetwork(learning_rate=learning_rate, model_type=60)]
        '''


    def train(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1,reshape=True)
        trainables = tf.trainable_variables()

        targetOps = updateTargetGraph(trainables, self.tau)
        rList = []
        #portfolio_list=[]
        total_steps = 0
        myBuffer = experience_buffer(self.buffer_size)
        episode_buffer = experience_buffer()
        e = self.startE

        stepDrop = (self.startE - self.endE) / self.anneling_steps
        with tf.Session() as sess:
            # 변수를 초기화한다.
            sess.run(init)
            if self.load_model == True:
                print('Loading Model...')
                # 모델을 불러온다
                ckpt = tf.train.get_checkpoint_state(self.path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                e = self.endE


            # 주요 신경망과 동일하게 타겟 신경망을 설정한다
            updateTarget(targetOps, sess)
            # 에피소드 시작
            for ii in range(self.num_episodes):
                rAll = 0
                d = False
                j = 0
                episode_buffer.buffer = []
                episode_reward_buffer = []
                self.environment.reset()
                self.agent.reset()
                rnn_state = np.array([mainQN.state_init for mainQN in self.mainQN])
                #print('%d 번째 episode 초기화 :' % ii,self.environment.idx, self.environment.KOSPI_idx, 'total num :',total_steps, '종목코드',self.environment.chart_code)
                s = [self.environment.get_image(days) for days in self.network_type]
                s_potfol = np.array(self.agent.get_states())

                episode_step = 1

                while j < self.max_epLength and not d:

                    j += 1

                    #입력값으로 행동선택하기(베이시안 + 볼트만)
                    all_Q_d = np.zeros([self.agent.NUM_ACTIONS])
                    before_rnn_state = rnn_state[:]
                    for i, mainQN in enumerate(self.mainQN):
                        Q_d, rnn_state[i] = sess.run([mainQN.Q_dist, mainQN.state_out], feed_dict={mainQN.inImage:[s[i]],
                                                                                                  mainQN.portfolio_state:[s_potfol],
                                                                                                  mainQN.state_in[0]:rnn_state[i][0],
                                                                                                  mainQN.state_in[1]:rnn_state[i][1],
                                                                                                  mainQN.temp:e,
                                                                                                  mainQN.keep_per:(1-e)+0.1,
                                                                                                  mainQN.phase:True})
                        all_Q_d += Q_d[0]
                    #모든 신경망의 확률값을 더한 뒤 나눔
                    #print(np.sum(all_Q_d))
                    all_Q_d /= len(self.network_type)
                    all_Q_d /= np.sum(all_Q_d)
                    #print(np.sum(all_Q_d))
                    a = np.random.choice(all_Q_d, p=all_Q_d)
                    action = np.argmax(all_Q_d == a)
                    #정책에 행동전달
                    delayed_reward = self.agent.act(action=action, confidence=all_Q_d[action])
                    d = self.environment.step()
                    if e > self.endE and total_steps > self.pre_train_steps:
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
                    s1 = [self.environment.get_image(days) for days in self.network_type]
                    s1_potfol = np.array(self.agent.get_states())

                    episode_reward_buffer.append(delayed_reward)
                    #버퍼에 저장
                    # 원래 버퍼 순서: 상태 액션 보상 다음상태 종료여부
                    # 수정 버퍼    : 현재이미지 액션 보상 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴
                    # 재수정 버퍼    : 현재이미지 액션 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴 보상(디스카운트 펙터설정)
                    #episode_buffer.add([s, action, delayed_reward, s1, s1_potfol, d, before_rnn_state, rnn_state, s_potfol  ]  )
                    episode_buffer.add([s, action, s1, s1_potfol, d, before_rnn_state, rnn_state, s_potfol])
                    if total_steps > self.pre_train_steps and total_steps % self.training_step == 0:
                        try:
                            #버퍼에서 데이터 가져오기
                            # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신

                            #원래 버퍼 순서: 상태 액션 보상 다음상태 종료여부
                            # 수정 버퍼    : 현재이미지 액션 보상 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴
                            # 배치 학습 데이터 크기
                            trainBatch, size = myBuffer.sample(self.replay_memory, rList)#(self.batch_size)
                            #print('훈련데이터 추출 결과 : ', trainBatch.shape)
                            #보상을 전행동에 영향이 가도록 할인인자로 곱해야함


                            for i in range(len(self.network_type)):
                                # 아래는 target Q-value를 업데이트하는 Double-DQN을 수행한다
                                # 주요 신경망에서 행동을 고른다.
                                #학습 시 베이시안과 볼트만을 사용하지 않음

                                #LSTM 학습을 위해서 랜덤한 에피소드에 랜덤한 날짜부터 replay memory만큼 선정하고 사용함
                                # 재수정 버퍼    : 현재이미지 액션 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴 보상(디스카운트 펙터설정)

                                feed_dict = {self.mainQN[i].inImage: [datas[i] for datas in trainBatch[:,2]],
                                                         self.mainQN[i].portfolio_state: [data for data in trainBatch[:,3]],
                                                         self.mainQN[i].state_in[0]: trainBatch[0,6][i][0],
                                                         self.mainQN[i].state_in[1]: trainBatch[0,6][i][1],
                                                         self.mainQN[i].keep_per: 1.0,
                                                         self.mainQN[i].phase: True}
                                Q1 = sess.run(self.mainQN[i].predict,
                                              feed_dict=feed_dict)
                                del feed_dict
                                feed_dict_2 = {self.targetQN[i].inImage: [datas[i] for datas in trainBatch[:,2]],
                                                         self.targetQN[i].portfolio_state: [data for data in trainBatch[:,3]],
                                                         self.targetQN[i].state_in[0]: trainBatch[0,6][i][0],
                                                         self.targetQN[i].state_in[1]: trainBatch[0,6][i][1],
                                                         self.targetQN[i].keep_per: 1.0,
                                                         self.targetQN[i].phase: True}
                                Q2 = sess.run(self.targetQN[i].Qout,  # feed_dict 수정해야함
                                              feed_dict=feed_dict_2)
                                del feed_dict_2
                                '''
                                Q1 = sess.run(self.mainQN[i].predict,
                                              feed_dict={self.mainQN[i].inImage: np.vstack(trainBatch[:, 3])})
                            
                                # 타겟 신경망에서 Q 값들을 얻는다.
                                Q2 = sess.run(self.targetQN[i].Qout, #feed_dict 수정해야함
                                              feed_dict={self.targetQN[i].inImage: np.vstack(trainBatch[:, 3])})
                            '''
                                # 종료 여부에 따라 가짜 라벨을 만들어준다
                                end_multiplier = -(trainBatch[:, 4] - 1)
                                # 타겟 신경망의 Q 값들 중에 주요 신경망에서 고른 행동 번째의 Q 값들을 가져온다.(이부분이 doubleQ)
                                doubleQ = Q2[range(size), Q1]
                                # 보상에 대한 더블 Q 값을 더해준다. y는 할인 인자
                                # targetQ 는 즉각적인 보상 + 다음 상태의 최대 보상(doubleQ)
                                targetQ = trainBatch[:, 8] + (self.y * doubleQ * end_multiplier)
                                # 우리의 타겟 값들과 함께 신경망을 업데이트해준다.
                                # 행동들에 대해서 targetQ 값과의 차이를 통해 손실을 구하고 업데이트
                                # 원래 버퍼 순서: 상태 액션 보상 다음상태 종료여부
                                # 수정 버퍼    : 현재이미지 액션 보상 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴

                                feed_dict = {self.mainQN[i].inImage: [datas[i] for datas in trainBatch[:, 0]],
                                             self.mainQN[i].portfolio_state: [data for data in trainBatch[:, 7]],
                                             self.mainQN[i].targetQ: targetQ,
                                             self.mainQN[i].actions: trainBatch[:, 1],
                                             self.mainQN[i].keep_per: 1.0,
                                             self.mainQN[i].state_in[0]: trainBatch[0, 5][i][0],
                                             self.mainQN[i].state_in[1]: trainBatch[0, 5][i][1],
                                             self.mainQN[i].phase: True
                                             }
                                _ = sess.run(self.mainQN[i].updateModel, \
                                             feed_dict=feed_dict)
                                del feed_dict
                                '''
                                _ = sess.run(self.mainQN[i].updateModel, \
                                             feed_dict={self.mainQN[i].inImage: np.vstack(trainBatch[:, 0]),
                                                        self.mainQN[i].targetQ: targetQ,
                                                        self.mainQN[i].actions: trainBatch[:, 1]})
                            '''
                            updateTarget(targetOps, sess)
                        except IndexError as e:
                            print(trainBatch)


                    rAll += delayed_reward
                    #rAll = delayed_reward
                    # 상태를 바꾼다.
                    del s
                    s = s1
                    del s_potfol
                    s_potfol = s1_potfol
                    total_steps += 1
                    episode_step += 1


                #portfolio_list.append(self.agent.portfolio_value)
                #할인인자 적용한 보상을 에피소드 버퍼에 추가
                accumulate = 0
                episode_reward_buffer.reverse()
                #print('%s episode_reward_len : ' % ii, len(episode_reward_buffer), 'episode_buffer_len :', len(episode_buffer.buffer))
                for i, reward in enumerate(episode_reward_buffer):
                    accumulate = self.discount_factor * accumulate + reward
                    idx = -(i+1)
                    episode_buffer.buffer[idx] += [accumulate]
                    #print(idx, len(episode_buffer.buffer[idx]))

                myBuffer.add(episode_buffer.buffer)
                if len(rList)+1>= self.buffer_size:
                    # self.buffer[0:1] = []
                    del rList[0]
                rList.append(rAll)
                self.environment.chartcode_value[self.environment.chart_code] += 1 if self.agent.portfolio_value > self.agent.initial_balance else -1
                print("%d %s %d %d %d %d" % (ii, self.environment.chart_code, rAll, self.agent.portfolio_value, self.agent.minimum_portfolio_value, self.agent.maximum_portfolio_value))
                #print("%d %4f %d %4f %4f %d %d"% (total_steps, np.mean(rList[-10:]), np.mean(portfolio_list), np.max(rList[-10:]),np.min(rList[-10:]),np.max(portfolio_list),np.min(portfolio_list)))#e)
                    #print(sys.getsizeof(myBuffer.buffer), sys.getsizeof(episode_buffer.buffer))
                #portfolio_list= []
                if total_steps > self.pre_train_steps and ii % 50 == 0:
                    try:
                        saver.save(sess, self.path + '/model-' + str(ii) + '.cptk')
                        with open('./value_chart.txt','w') as f:
                            data = json.dumps(self.environment.chartcode_value)
                            f.write(data)
                        del data
                        #print("Saved Model")
                    except:
                        pass
                sleep(2)
            # 학습 끝 평균 보상을 표시
            saver.save(sess, self.path + '/model-' + str(ii) + '.cptk')
            print("평균 episode 별 보상 값 : " + str(sum(rList) / self.num_episodes) )
    def test(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        trainables = tf.trainable_variables()

        targetOps = updateTargetGraph(trainables, self.tau)
        rList = []
        total_steps = 0
        myBuffer = experience_buffer()
        episode_buffer = experience_buffer()
        e = self.startE

        stepDrop = (self.startE - self.endE) / self.anneling_steps
        with tf.Session() as sess:
            if self.load_model == True:
                print('Loading Model...')
                # 모델을 불러온다
                ckpt = tf.train.get_checkpoint_state(self.path)
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            # 변수를 초기화한다.
            sess.run(init)
            # 주요 신경망과 동일하게 타겟 신경망을 설정한다
            updateTarget(targetOps, sess)
            # 에피소드 시작
            for ii in range(self.num_episodes):
                rAll = 0
                d = False
                j = 0
                experience_buffer.buffer = []
                self.environment.reset()
                self.agent.reset()
                rnn_state = np.array([mainQN.state_init for mainQN in self.mainQN])
                print('%d 번째 episode 초기화 :' % ii, self.environment.idx, self.environment.KOSPI_idx, 'total num :',
                      total_steps, '종목코드', self.environment.chart_code)
                s = [self.environment.get_image(days) for days in self.network_type]
                s_potfol = np.array(self.agent.get_states())
                while j < self.max_epLength and not d:

                    j += 1

                    # 입력값으로 행동선택하기(베이시안 + 볼트만)
                    all_Q_d = np.zeros([self.agent.NUM_ACTIONS])
                    before_rnn_state = rnn_state[:]
                    for i, mainQN in enumerate(self.mainQN):
                        Q_d, rnn_state[i] = sess.run([mainQN.Q_dist, mainQN.state_out],
                                                     feed_dict={mainQN.inImage: [s[i]],
                                                                mainQN.portfolio_state: [s_potfol],
                                                                mainQN.state_in[0]: rnn_state[i][0],
                                                                mainQN.state_in[1]: rnn_state[i][1],
                                                                mainQN.temp: e,
                                                                mainQN.keep_per: (1 - e) + 0.1,
                                                                mainQN.phase: True})
                        all_Q_d += Q_d[0]
                    # 모든 신경망의 확률값을 더한 뒤 나눔
                    # print(np.sum(all_Q_d))
                    all_Q_d /= len(self.network_type)
                    all_Q_d[0] += 1 - np.sum(all_Q_d)
                    # print(np.sum(all_Q_d))
                    a = np.random.choice(all_Q_d, p=all_Q_d)
                    action = np.argmax(all_Q_d == a)
                    # 정책에 행동전달
                    immediate_reward, delayed_reward = self.agent.act(action=action, confidence=all_Q_d[action])

                    if e > self.endE and total_steps > self.pre_train_steps:
                        e -= stepDrop

                    if delayed_reward == 0 and total_steps % 5 == 0:
                        delayed_reward = immediate_reward
                        self.agent.base_portfolio_value = self.agent.portfolio_value
                    # 다음 인덱스로 넘어가기
                    d = self.environment.step()
                    # 다음이미지,포폴 받기
                    # print('total step :', total_steps, 'current episode step : ', j, 'idx :', self.environment.idx, 'kospi_idx', self.environment.KOSPI_idx, '종목코드',self.environment.chart_code)
                    s1 = [self.environment.get_image(days) for days in self.network_type]
                    s1_potfol = np.array(self.agent.get_states())
                    # 버퍼에 저장
                    # 원래 버퍼 순서: 상태 액션 보상 다음상태 종료여부
                    # 수정 버퍼    : 현재이미지 액션 보상 다음이미지, 다음포폴상태 종료여부 이전상태LSTM, 상태LSTM 현재포폴


                    rAll += delayed_reward
                    # 상태를 바꾼다.
                    del s
                    s = s1
                    del s_potfol
                    s_potfol = s1_potfol
                    total_steps += 1
                if total_steps > self.pre_train_steps and ii % 50 == 0:
                    saver.save(sess, self.path + '/model-' + str(ii) + '.cptk')
                    print("Saved Model")
                    sleep(3)
                rList.append(rAll)
                myBuffer.add(episode_buffer.buffer)
                if len(rList) % 10 == 0:
                    print(total_steps, np.mean(rList[-10:]), e)


                sleep(2)
            # saver.save(sess, self.path + '/model-' + str(i) + '.cptk')
            # 평균 보상을 표시
            print("평균 episode 별 보상 값 : " + str(sum(rList) / self.num_episodes))


if __name__ == "__main__":
    obj = PolicyLearner()
    obj.train()

















