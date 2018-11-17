
import numpy as np
import torch as th
import pickle
from world import world
from dqn_buyer import DQN_buyer
from dqn_seller import DQN_seller
# do not render the scene2

nSellers = 4
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
Num_action = 2
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

buyerBots = []
sellerBots = []
for i in range(nSellers):
    buyerBots.append(dqrnAgent('BuyerAgent'+str(i)))
    sellerBots.append(dqrnAgent('SellerAgent'+str(i)))


for i_episode in range(n_episode):
    obs_seller, obs_buyer = world.reset()
    obs_seller = np.stack(obs_seller)
    obs_seller = obs_seller.flatten()
    obs_buyer = np.stack(obs_buyer)
    if i_episode % 50 == 0:
        print('------------------------------------')
        print('Initial Conditions:')
        print(obs_seller)
        print(obs_buyer)
    total_reward = 0.0
    for t in range(max_steps):
        
        #get buyer actions
        actions_seller = [0]*nSellers
        for i in range(nSellers):
            action = sellerBots[i].choose_action(obs_seller)
            actions_seller[i] = action
        
        #get buyer actions
        actions_buyer = [0]*nSellers
        for i in range(nSellers):
            action = buyerBots[int(obs_buyer[i][3])].choose_action(obs_buyer[i]) #obs_buyer[i][3] to get the deter of bot
            actions_buyer[i] = action
        
        
        #step
        obs_seller_, obs_buyer_, rewards_seller, rewards_buyer = world.step(actions_seller, actions_buyer)
        obs_seller_ = obs_seller_.flatten()
        
        #build coordination for sellers
        avg_reward = sum(rewards_seller)/len(rewards_seller)
        rewards_seller = [x+avg_reward for x in rewards_seller]
        
        #buyerbot updates
        for i in range(nSellers):
            buyerBots[int(obs_buyer[i][3])].store_transition(obs_buyer[i], actions_buyer[i], rewards_buyer[i], obs_buyer_[i])
            if buyerBots[int(obs_buyer[i][3])].memory_counter > MEMORY_CAPACITY:
                buyerBots[int(obs_buyer[i][3])].learn()
            obs_buyer[i] = obs_buyer_[i]

        #sellerbot updates
        for i in range(nSellers):
            sellerBots[i].store_transition(obs_seller, actions_seller[i], rewards_seller[i], obs_seller_)
            if sellerBots[i].memory_counter > MEMORY_CAPACITY:
                sellerBots[i].learn()
        obs_seller = obs_seller_
            
        
    if i_episode % 50 == 0:    
        print('Episode: %d, reward_seller = %f, reward_buyer = %f' % (i_episode, max(rewards_seller), max(rewards_buyer)))
    
#    if(average_reward > 0 and max(rewards_buyer) > 0):
#        input('press enter')


    
    
    if i_episode > episodes_before_train:
        
        #too competitive bots saved
#        if max(rewards_seller) > maxSellerReward and max(rewards_seller) >0:
#            th.save(sellerBots[rewards_seller.index(max(rewards_seller))], 'sellerSavedModels/sellerBot-'+str(i_episode)+'.model')
#            maxSellerReward = max(rewards_seller)
#            print('SELLER SAVED')
#            for i in range(nSellers):
#                sellerBots[i] = th.load('sellerSavedModels/sellerBot-'+str(i_episode)+'.model')
#  
#        if max(rewards_buyer) > maxBuyerReward and max(rewards_buyer) >0:
#            th.save(buyerBots[rewards_buyer.index(max(rewards_buyer))], 'buyerSavedModels/buyerBot-'+str(i_episode)+'.model')
#            maxBuyerReward = max(rewards_buyer)
#            print('BUYER SAVED')
#            #copy this buyer to all buyers - DUBIOUS
#            for i in range(nSellers):
#                buyerBots[i] = th.load('buyerSavedModels/buyerBot-'+str(i_episode)+'.model')
                    

        #compromising bots saved
#        RewardSum = max(rewards_seller) + max(rewards_buyer)
#        
#        if RewardSum > maxRewardSum and max(rewards_seller) > 0 and max(rewards_buyer) > 0:
#            th.save(sellerBots[rewards_seller.index(max(rewards_seller))], 'sellerSavedModels/sellerBot-'+str(i_episode)+'.model')
#            print('SELLER SAVED')
#            th.save(buyerBots[0], 'buyerSavedModels/buyerBot0-'+str(i_episode)+'.model')
#            th.save(buyerBots[1], 'buyerSavedModels/buyerBot1-'+str(i_episode)+'.model')
#            th.save(buyerBots[2], 'buyerSavedModels/buyerBot2-'+str(i_episode)+'.model')
#  
#            print('BUYER SAVED')
#            maxRewardSum = RewardSum
                   
        if i_episode % 50 == 0:        
            print('DONE')
            for i in range(nSellers):
                print('Case ' +str(i))
                print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))
            sellerRewardArr.append(sum(rewards_seller))
            buyerRewardArr.append(sum(rewards_buyer))
            
                
    if i_episode%1000 == nSellers+3:
        th.save(sellerBots[rewards_seller.index(max(rewards_seller))], 'sellerSavedModels/sellerBot-'+str(i_episode)+'.model')
        for i in range(nSellers):
            th.save(buyerBots[i], 'buyerSavedModels/buyerBot'+str(i)+'-'+str(i_episode)+'.model')
      
    with open('sellerReward.pickle', 'wb') as handle:
        pickle.dump(sellerRewardArr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('buyerReward.pickle', 'wb') as handle:
        pickle.dump(buyerRewardArr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    
    
    
    
    
    
    
    