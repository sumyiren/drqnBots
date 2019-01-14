
import numpy as np
import random
import tensorflow as tf
from drqnMLEngine.world import world
from drqnMLEngine.DRQNbuyer import dqrnBuyer
from drqnMLEngine.DRQNseller import dqrnSeller


class Trainer(object):

    def __init__(self, args):
        self.nSellers = 2
        self.reward_record = []

        self.n_agents = self.nSellers
        self.n_states = 3 #FIXTHIS!!!
        self.n_actions = 3
        self.maxSellerReward = -100
        self.maxBuyerReward = -100
        self.maxRewardSum = -100
        self.max_steps = 150
        self.n_episode = 4050

        # Parameter setting
        self.Num_action = 3
        self.Gamma = 0.99
        self.Learning_rate = 0.00025
        self.Epsilon = 0.1
        self.Final_epsilon = 0.01

        self.Num_replay_memory = 2000000
        self.Num_start_training = 100000
        self.Num_training = self.n_episode*self.max_steps

        # DRQN Parameters
        self.step_size = 50

        self.world = world(self.nSellers, self.max_steps)
        self.bB = []
        self.sBA = None
        self.count = 0
        self.save_folder = args.job_dir
        self.sess = None
        self.saver = None
        self.Num_batch = 8
        self.episode_no = 0
        self.isRandom = True

    def flattenList(self, list):
        flat_list = [item for sublist in list for item in sublist]
        return flat_list

    def resetWorld(self, world):
        obs_seller, obs_buyer = self.world.reset()
        obs_seller = np.stack(obs_seller)
        obs_seller = self.flattenList(obs_seller)
        obs_buyer = np.stack(obs_buyer)
        return obs_seller, obs_buyer


    def performMiniBatching(self, sbB):
        episode_batch = random.sample(sbB.Replay_memory, sbB.Num_batch)

        sbB.minibatch = []
        sbB.batch_end_index = []
        sbB.count_minibatch = 0

        for episode_ in episode_batch:
            episode_start = np.random.randint(0, len(episode_) + 1 - self.step_size)
            for step_ in range(self.step_size):
                sbB.minibatch.append(episode_[episode_start + step_])
                if step_ == self.step_size - 1:
                    sbB.batch_end_index.append(sbB.count_minibatch)

                sbB.count_minibatch += 1

        # Save the each batch data
        observation_batch      = [batch[0] for batch in sbB.minibatch]
        action_batch           = [batch[1] for batch in sbB.minibatch]
        reward_batch           = [batch[2] for batch in sbB.minibatch]
        observation_next_batch = [batch[3] for batch in sbB.minibatch]
        terminal_batch            = [batch[4] for batch in sbB.minibatch]


        # Get y_prediction
        sbB.y_batch = []
        sbB.action_in = []
        Q_batch = sbB.get_output_batch(observation_next_batch, sbB.Num_batch, self.step_size)
        #            Q_batch = output.eval(feed_dict = {x: observation_next_batch, rnn_batch_size: Num_batch, rnn_step_size: step_size})
        for count, j in enumerate(sbB.batch_end_index):
            sbB.action_in.append(action_batch[j])
            if terminal_batch[j] == True:
                sbB.y_batch.append(reward_batch[j])
            else:
                sbB.y_batch.append(reward_batch[j] + self.Gamma * np.max(Q_batch[count]))

        sbB.trainStep(sbB.action_in, sbB.y_batch, observation_batch, sbB.Num_batch, self.step_size)
        # Reduce epsilon at training mode


    def saveModel(self, step):
        print('SAVED MODEL')
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, self.save_folder+'/model', global_step=step)


    def saveExperience(self, sbB):
        # Save experience to the Replay memory
        sbB.episode_memory.append([sbB.observation, sbB.action, sbB.reward, sbB.observation_next, sbB.terminal])

        if len(sbB.Replay_memory) > self.Num_replay_memory:
            del sbB.Replay_memory[0]

        # Update parameters at every iteration
        sbB.step += 1
        sbB.score += sbB.reward

        # if self.bB[0].step > self.Num_start_training:
        #     if (sbB.step-self.Num_start_training)%(self.max_steps*1) == 0: #every 100 episodes
        #         print('SAVED at ' + str(sbB.step))
        #         sbB.saveModel(self.save_folder, sbB.step, i)


        sbB.observation = sbB.observation_next

        sbB.observation_set.append(sbB.observation)

        if len(sbB.observation_set) > self.step_size:
            del sbB.observation_set[0]

    def startTraining(self):
        #initialize bots envs
        obs_seller, obs_buyer = self.resetWorld(world)
        for i in range(self.nSellers):
            self.bB.append(dqrnBuyer('BuyerAgent'+str(i)))
            self.bB[i].build_model()
            self.bB[i].observation = obs_buyer[i]
            self.bB[i].action = self.world.action_space.sample()

        self.sBA = dqrnSeller('SellerAgent')
        self.sBA.build_model()
        self.sBA.observation = obs_seller
        self.sBA.action = [self.world.action_space.sample()]*self.nSellers

        #run session
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        #actions bundled up for world
        actions_buyer = [0]*self.nSellers

        for i in range(self.nSellers):
            actions_buyer[i] = self.bB[i].action
        actions_seller = self.sBA.action

        obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
            = self.world.step(actions_seller, actions_buyer)

        obs_seller_ = self.flattenList(obs_seller_)

        for i in range(self.nSellers):
            self.bB[i].observation = obs_buyer_[i]
            self.bB[i].reward = rewards_buyer[i]
            self.bB[i].terminal = done
        self.sBA.observation = obs_seller_
        self.sBA.reward = np.average(rewards_seller)
        self.sBA.terminal = done

        while True:

            if self.bB[0].step <= self.Num_start_training:
                state = 'Observing'

                for i in range(self.nSellers):
                    self.bB[i].action = np.zeros([self.Num_action])
                    self.bB[i].action[random.randint(0, self.Num_action - 1)] = 1.0
                    action_step = np.argmax(self.bB[i].action)
                    actions_buyer[i] = action_step

                self.sBA.action = []
                for i in range(self.nSellers):
                    sBA_pseudo_action = np.zeros([self.Num_action])
                    sBA_pseudo_action[random.randint(0, self.Num_action - 1)] = 1.0
                    self.sBA.action.extend(sBA_pseudo_action)
                    action_step = np.argmax(sBA_pseudo_action)
                    actions_seller[i] = action_step


                obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
                    = self.world.step(actions_seller, actions_buyer)
                obs_seller_ = self.flattenList(obs_seller_)

                for i in range(self.nSellers):
                    self.bB[i].observation_next = obs_buyer_[i]
                    self.bB[i].reward = rewards_buyer[i]
                    self.sBA.terminal, self.bB[i].terminal = done, done
                    if self.bB[i].step % 100 == 0:
                        print('step: ' + str(self.bB[i].step) + ' / '  + 'state: ' + state)
                self.sBA.observation_next = obs_seller_
                self.sBA.reward = np.average(rewards_seller)

            elif not self.bB[0].terminal:
                
                # Training
                state = 'Training'
                # if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value
#                if random.random() < self.Epsilon:
                if self.isRandom:
                    for i in range(self.nSellers):
                        self.bB[i].action = np.zeros([self.Num_action])
                        self.bB[i].action[random.randint(0, self.Num_action - 1)] = 1.0
                        action_step = np.argmax(self.bB[i].action)
                        actions_buyer[i] = action_step

                    self.sBA.action = []
                    for i in range(self.nSellers):
                        sBA_pseudo_action = np.zeros([self.Num_action])
                        sBA_pseudo_action[random.randint(0, self.Num_action - 1)] = 1.0
                        self.sBA.action.extend(sBA_pseudo_action)
                        action_step = np.argmax(sBA_pseudo_action)
                        actions_seller[i] = action_step
                else:
                    for i in range(self.nSellers):
                        Q_value = self.bB[i].get_output(self.bB[i].observation_set, self.Num_batch, self.step_size)
                        self.bB[i].action = np.zeros([self.Num_action])
                        self.bB[i].action[np.argmax(Q_value)] = 1
                        action_step = np.argmax(self.bB[i].action)
                        actions_buyer[i] = action_step

                    Q_value = self.sBA.get_output(self.sBA.observation_set, self.Num_batch, self.step_size)
                    self.sBA.action = []
                    for i in range(self.nSellers):
                        #                sBA.action[i] = np.argmax(Q_value[i:i+3])
                        Q_value_pseudo = Q_value[i:i+3]
                        sBA_pseudo_action = np.zeros([self.Num_action])
                        sBA_pseudo_action[np.argmax(Q_value_pseudo)] = 1
                        self.sBA.action.extend(sBA_pseudo_action)
                        action_step = np.argmax(sBA_pseudo_action)
                        actions_seller[i] = action_step

                obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
                    = self.world.step(actions_seller, actions_buyer)
                obs_seller_ = self.flattenList(obs_seller_)
                for i in range(self.nSellers):
                    self.bB[i].observation_next = obs_buyer_[i]
                    self.bB[i].reward = rewards_buyer[i]
                    self.sBA.terminal, self.bB[i].terminal = done, done
                self.sBA.observation_next = obs_seller_
                self.sBA.reward = np.average(rewards_seller)


                for i in range(self.nSellers):
                    self.performMiniBatching(self.bB[i])
                self.performMiniBatching(self.sBA)
                
#                if self.Epsilon > self.Final_epsilon:
#                    self.Epsilon -= 1.0/self.Num_training
                
            
            # Save experience to the Replay memory
            for i in range(self.nSellers):
                self.saveExperience(self.bB[i])
            self.saveExperience(self.sBA)


            # Terminal
            if self.bB[0].terminal:
                if self.count % 1 == 0:
                    print('DONE')
                    for i in range(self.nSellers):
                        print('Case ' +str(i))
                        print('SellerAsk = ' +str(obs_buyer_[i][0])+ 'BuyerAsk = ' + str(obs_buyer_[i][1]))

                # # Run Saver here
                # if self.bB[0].step > self.Num_start_training:

                if self.count % 500 == 0 and self.bB[0].step > self.Num_start_training:
                    self.saveModel(self.count)

                obs_seller, obs_buyer = self.resetWorld(world)

                if self.count % 1 == 0:
                    print('------------------------------------')
                    print('Initial Conditions:' + str(self.count))
                    print(obs_seller)
                    print(obs_buyer)
                self.count = self.count + 1
                for i in range(self.nSellers):

                    if len(self.bB[i].episode_memory) > self.step_size:
                        self.bB[i].Replay_memory.append(self.bB[i].episode_memory)
                    self.bB[i].episode_memory = []

                    self.bB[i].score = 0
                    self.bB[i].episode += 1
                    self.bB[i].observation = obs_buyer[i]

                    self.bB[i].observation_set = []
                    for j in range(self.step_size):
                        self.bB[i].observation_set.append(self.bB[i].observation)
                    self.bB[i].terminal = False


                if len(self.sBA.episode_memory) > self.step_size:
                    self.sBA.Replay_memory.append(self.sBA.episode_memory)
                self.sBA.episode_memory = []

                self.sBA.score = 0
                self.sBA.episode += 1
                self.sBA.observation = obs_seller

                self.sBA.observation_set = []
                for j in range(self.step_size):
                    self.sBA.observation_set.append(self.sBA.observation)

                self.sBA.terminal = False
                
                self.isRandom = random.random() < self.Epsilon
                
                #check ending here
                if self.episode_no > self.n_episode:
                    break
                if self.bB[0].step > self.Num_start_training:
                    self.episode_no = self.episode_no + 1
                
                
        print('FINISHED')
                





