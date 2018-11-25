
import numpy as np
import torch as th
import pickle
import random
from world import world
from DRQN import dqrnAgent

nSellers = 1
maxSellers = 5 #max number of sellers this is trained for
reward_record = []
np.random.seed(1234)
th.manual_seed(1234)

n_agents = nSellers
n_states = 3 #FIXTHIS!!!
n_actions = 3
maxSellerReward = -100
maxBuyerReward = -100
maxRewardSum = -100
max_steps = 150
n_episode = 60100

# Parameter setting
Num_action = 3
Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1
Final_epsilon = 0.01

Num_replay_memory = 200
Num_start_training = 1000
Num_training = 2000
Num_testing  = 10000
Num_update = 250
Num_batch = 8
Num_episode_plot = 30

# DRQN Parameters
step_size = 4
lstm_size = 256
flatten_size = 4

world = world(nSellers, max_steps)
sellerRewardArr= []
buyerRewardArr = []

bB = []
sB = []


def flattenList(list):
    flat_list = [item for sublist in list for item in sublist]
    return flat_list

def resetWorld(world):
    obs_seller, obs_buyer = world.reset()
    obs_seller = np.stack(obs_seller)
    obs_seller = flattenList(obs_seller)
    obs_buyer = np.stack(obs_buyer)
    return obs_seller, obs_buyer


def performMiniBatching(sbB):
    episode_batch = random.sample(sbB.Replay_memory, Num_batch)

    sbB.minibatch = []
    sbB.batch_end_index = []
    sbB.count_minibatch = 0

    for episode_ in episode_batch:
        episode_start = np.random.randint(0, len(episode_) + 1 - step_size)
        for step_ in range(step_size):
            sbB.minibatch.append(episode_[episode_start + step_])
            if step_ == step_size - 1:
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
    Q_batch = sbB.get_output_batch(observation_next_batch, Num_batch, step_size)
    #            Q_batch = output.eval(feed_dict = {x: observation_next_batch, rnn_batch_size: Num_batch, rnn_step_size: step_size})

    for count, j in enumerate(sbB.batch_end_index):
        sbB.action_in.append(action_batch[j])
        if terminal_batch[i] == True:
            sbB.y_batch.append(reward_batch[j])
        else:
            sbB.y_batch.append(reward_batch[j] + Gamma * np.max(Q_batch[count]))

    sbB.trainStep(sbB.action_in, sbB.y_batch, observation_batch, Num_batch, step_size)
    # Reduce epsilon at training mode
    if sbB.epsilon > Final_epsilon:
        sbB.epsilon -= 1.0/Num_training



def saveExperience(sbB):
    # Save experience to the Replay memory
    sbB.episode_memory.append([sbB.observation, sbB.action, sbB.reward, sbB.observation_next, sbB.terminal])

    if len(sbB.Replay_memory) > Num_replay_memory:
        del sbB.Replay_memory[0]

    # Update parameters at every iteration
    sbB.step += 1
    sbB.score += sbB.reward

    # if sbB.step%5000 == 0:
    #     print('SAVED')
    #     sbB.saveModel(5000)


    sbB.observation = sbB.observation_next

    sbB.observation_set.append(sbB.observation)

    if len(sbB.observation_set) > step_size:
        del sbB.observation_set[0]


#initialize bots envs
obs_seller, obs_buyer = resetWorld(world)
for i in range(nSellers):
    bB.append(dqrnAgent('BuyerAgent'+str(i)))
    sB.append(dqrnAgent('SellerAgent'+str(i)))
    bB[i].observation = obs_buyer[i]
    sB[i].observation = obs_seller
    bB[i].action = world.action_space.sample()
    sB[i].action = world.action_space.sample()

#actions bundled up for world
actions_seller = [0]*nSellers
actions_buyer = [0]*nSellers

for i in range(nSellers):
    actions_buyer[i] = bB[i].action
    actions_seller[i] = sB[i].action


obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
    = world.step(actions_seller, actions_buyer)

obs_seller_ = flattenList(obs_seller_)

for i in range(nSellers):
    bB[i].observation = obs_buyer_[i]
    sB[i].observation = obs_seller_
    bB[i].reward, sB[i].reward = rewards_buyer[i], rewards_seller[i]
    sB[i].terminal, bB[i].terminal = done, done


while True:

    if bB[0].step <= Num_start_training:
        state = 'Observing'

        for i in range(nSellers):
            bB[i].action, sB[i].action = np.zeros([Num_action]), np.zeros([Num_action])
            bB[i].action[random.randint(0, Num_action - 1)] = 1.0
            sB[i].action[random.randint(0, Num_action - 1)] = 1.0
            action_step = np.argmax(bB[i].action)
            actions_buyer[i] = action_step
            action_step = np.argmax(sB[i].action)
            actions_seller[i] = action_step


        obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
            = world.step(actions_seller, actions_buyer)
        obs_seller_ = flattenList(obs_seller_)

        for i in range(nSellers):
            bB[i].observation_next = obs_buyer_[i]
            sB[i].observation_next = obs_seller_
            bB[i].reward -= 5 * abs(bB[i].observation_next[0])
            sB[i].reward -= 5 * abs(sB[i].observation_next[0])
            if bB[i].step % 100 == 0:
                print('step: ' + str(bB[i].step) + ' / '  + 'state: ' + state)

    elif bB[0].step <= Num_start_training + Num_training:
        # Training
        state = 'Training'

        # if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value
        if random.random() < Epsilon:
            for i in range(nSellers):
                bB[i].action, sB[i].action = np.zeros([Num_action]), np.zeros([Num_action])
                bB[i].action[random.randint(0, Num_action - 1)] = 1.0
                sB[i].action[random.randint(0, Num_action - 1)] = 1.0
                action_step = np.argmax(bB[i].action)
                actions_buyer[i] = action_step
                action_step = np.argmax(sB[i].action)
                actions_seller[i] = action_step

        else:
            for i in range(nSellers):
                Q_value = bB[i].get_output(bB[i].observation_set, 1, step_size)
                bB[i].action = np.zeros([Num_action])
                bB[i].action[np.argmax(Q_value)] = 1
                action_step = np.argmax(bB[i].action)
                actions_buyer[i] = action_step

                Q_value = sB[i].get_output(sB[i].observation_set, 1, step_size)
                sB[i].action = np.zeros([Num_action])
                sB[i].action[np.argmax(Q_value)] = 1
                action_step = np.argmax(sB[i].action)
                actions_seller[i] = action_step

            obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
                    = world.step(actions_seller, actions_buyer)
            obs_seller_ = flattenList(obs_seller_)

            for i in range(nSellers):
                bB[i].observation_next = obs_buyer_[i]
                sB[i].observation_next = obs_seller_
                bB[i].reward -= 5 * abs(bB[i].observation_next[0])
                sB[i].reward -= 5 * abs(sB[i].observation_next[0])


            for i in range(nSellers):
                performMiniBatching(bB[i])
                performMiniBatching(sB[i])


    # Save experience to the Replay memory
    for i in range(nSellers):
        saveExperience(bB[i])
        saveExperience(sB[i])

    # Terminal
    if bB[0].terminal == True:
        obs_seller, obs_buyer = resetWorld(world)
        for i in range(nSellers):
            print('step: ' + str(bB[i].step) + ' / ' + 'episode: ' + str(bB[i].episode) + ' / ' + 'state: ' + state  + ' / '  + 'epsilon: ' + str(bB[i].epsilon) + ' / '  + 'score: ' + str(bB[i].score))

            if len(bB[i].episode_memory) > step_size:
                bB[i].Replay_memory.append(bB[i].episode_memory)
            bB[i].episode_memory = []

            bB[i].score = 0
            bB[i].episode += 1
            bB[i].observation = obs_buyer[i]

            bB[i].observation_set = []
            for j in range(step_size):
                bB[i].observation_set.append(bB[i].observation)

            print('step: ' + str(sB[i].step) + ' / ' + 'episode: ' + str(sB[i].episode) + ' / ' + 'state: ' + state  + ' / '  + 'epsilon: ' + str(sB[i].epsilon) + ' / '  + 'score: ' + str(sB[i].score))

            if len(sB[i].episode_memory) > step_size:
                sB[i].Replay_memory.append(sB[i].episode_memory)
            sB[i].episode_memory = []

            sB[i].score = 0
            sB[i].episode += 1
            sB[i].observation = obs_seller

            sB[i].observation_set = []
            for j in range(step_size):
                sB[i].observation_set.append(sB[i].observation)





