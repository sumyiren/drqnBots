from testClass import testClass
import pickle

test = testClass()
dealCounter = 0
numTests = 20

startArrays = []
endArrays = []

def checkDealMade(obs_seller):
    dealsMade = False
    for i in range(test.nSellers):
        if abs(obs_seller[0][i*2]-obs_seller[0][i*2+1]) <= 1:
            dealsMade = True
    return dealsMade
            
        
for j in range(numTests):
    print('Testing '+str(j))
    done = False
    obs_seller, obs_buyer = test.restart()
    startArrays.append({'seller':obs_seller, 'buyer':obs_buyer})
    while not done:
        obs_seller, obs_buyer, done = test.stepAction()
        if done:
            endArrays.append({'seller':obs_seller, 'buyer':obs_buyer})
            
with open('./testResults/multitestdqrn20.pickle', 'wb') as handle:
    pickle.dump({'start':startArrays, 'end':endArrays}, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('filename.pickle', 'rb') as handle:
#    b = pickle.load(handle)        
#print('Total Deals out of ' + str(numTests) + ': ' + str(dealCounter))