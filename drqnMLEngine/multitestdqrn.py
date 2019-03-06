from testClass import testClass


test = testClass()
dealCounter = 0
numTests = 100
def checkDealMade(obs_seller):
    dealsMade = False
    for i in range(test.nSellers):
        if abs(obs_seller[0][i*2]-obs_seller[0][i*2+1]) <= 1:
            dealsMade = True
    return dealsMade
            
        
for j in range(numTests):
    print('Testing '+str(j))
    done = False
    test.restart()
    while not done:
        obs_buyer, obs_seller, done = test.stepAction()
        if done:
            if checkDealMade(obs_seller):
                dealCounter += 1
            
        
print('Total Deals out of ' + str(numTests) + ': ' + str(dealCounter))