'''
AIFR by coupled auto-encoder network

http://chenfeixu.com/wp-content/uploads/2014/04/CAN.pdf


See experiments/experiment_d/README.md for details
'''

# imports the libraries 

# defines the classes

class coupledAutoModel():
    def __init__(self):
        super(CoupledAutoModel, self).__init__()

        # young architecture

        '''Encoder'''

        '''Decoder'''

        # old architecture
        
        '''Encoder'''

        '''Decoder'''

    def initWeights(self):
        # w1

        # b1

        # w1_bar

        # c1

        
        # w2

        # b2

        # w2_bar

        # c2

    def forward(self, x1, x2):
        x1 = pass # encode young x1
        x1 = pass # sigmoid young x1
        x1_bar = pass # decode young x1_bar

        x2 = pass # endode old x2
        x2 = pass # sigmoid old x2
        x2_bar = pass # decode old x2_bar

        c1 = pass # compute c1 young bias
        c2 = pass # compute c2 old bias

def coupledAutoModel_(**kwargs):
    model_auto_encoder = coupledAutoModel
    return model_auto_encoder

class bridgeModel():
    def __init__(self):
        super(bridgeModel, self).__init__()

        '''Age Part'''
        # x1, both wv1 and bv1 are automatically calculated
        pass
        
        # after applying this to x1, you get A1(i)
        # used for x2, wv2 and bv2 are automatically calculated
        pass

        # no of neurons is 800
        # aging bridge network from A1 to A2 hat
        pass

        # deaging bridge network from A2 to A1 hat
        pass

        '''Identity Part'''
        pass

        '''Reconstructs x1_bar and x2_bar'''
        # age_young
        pass

        # age_old
        pass

        self.initWeights()

    def initWeights(self):
        # Ha1
        pass

        # Ha2
        pass

        # ba1
        pass

        # ba2
        pass

        # Hd1
        pass

        # Hd2
        pass

        # bd1
        pass

        # bd2
        pass

    def forward(self, x1, x2, c1, c2): # substitue all self.pass
        '''Age Part'''
        # aging part, goes from x1 to a1 to a2_hat
        x1_base = x1
        x1 = pass # encode x1
        a1 = pass # sigmoid of x1, gives a1(i)

        by1 = pass # bridge young 1
        by1 = pass # sigmoid of by1
        by2 = pass # bridge young 2
        by2 = pass # sigmoid of by2

        # deaging part, goes from x2 to a2 to a1_hat
        x2_base = x2
        x2 = pass # encode x2
        a2 = pass # sigmoid of x2, gives a2(i)

        bo1 = pass # bridge old 1
        bo1 = pass # sigmoid of bo1
        bo2 = # bridge old 2
        a1_hat = # sigmoid of bo2

        '''Identity Part'''
        id1 = pass # x1_base linear identity 
        id1 = pass # sigmoid of id1 
        id2 = pass # x2_base linear identity 
        id2 = pass # sigmoid of id2

        wu1 = pass # linear identity young weight 
        wu2 = pass # linear identity old weight 
        bu1 = pass # linear identity young bias 
        bu2 = pass # linear identity old bias 

        wv1 = pass # endode x1.weight 
        wv2 = pass # endode x2.weight

        '''Addition Part'''
        wv1_hat = pass # reconstructed x1_age.weight
        wu1_hat = pass # reconstructed x1_identity.weight
        wv2_hat = pass # reconstructed x2_age.weight
        wu2_hat = pass # reconstructed x2_identity.weight

        mul1_x1 = pass # matmul(wv1_hat, a1_hat)
        mul2_x1 = pass # matmul(wu1_hat, id1)

        mul1_x1 = pass # mul1_x1()
        mul2_x1 = pass # mul2_x1()

        add_mull = pass # add layer (mul1_x1, 1, mul2_x1)
        add_x1 = pass # add layer (add_mul1, 1, c1)

        reconstruct_x1 = pass # sigmoid of add_x1
        mul1_x2 = pass # matmul(wv2_hat, a2_hat())
        mul2_x2 = pass # matmul(wu2_hat, id2())

        mul1_x2 = pass # mul1_x2()
        mul2_x2 = pass # mul2_x2()

        add_mul2 = pass # add layer (mul1_x2, 1, mul2_x2)
        add_x2 = pass # add layer (add_mul2, 1, c2)

        reconstruct_x2 = pass # sigmoid of add_x2

        return a1, a1_hat, a2, a2_hat, id1, id2, x1_base, reconstruct_x1, x2_base, reconstruct_x2, wu1, wu2, wv1, wv2, bu1, bu2
    
    def bridgeModel_(**kwargs):
        bridgeModel = bridgeModel 
        return bridgeModel
    
class testModelYng():
    def __init__(self):
        super(testModelYng, self).__init__()
        
        self.initWeights()
    
    def initWeights(self):
        pass

    def forward(self, x1):
        x1 = pass # encode of x1
        x1 = pass # sigmoid of x1

        features_yng_weight = pass # linear encode of yng.weight
        features_yng_bias = pass # linear encode of yng.bias

        return x1, features_yng_weight, features_yng_bias # feature vector of young image

class testModelOld():
    def __init__(self):
        super(testModelOld, self).__init__()

        self.initWeights()
    
    def initWeights(self):
        pass

    def forward(self, x2):
        x2 = pass # encode of x2
        x2 = pass # sigmoid of x2

        features_old_weight = pass # linear encode of old.weights
        features_old_bias = pass # linear encode of old.bias

        return x2, features_old_weight, features_old_bias # feature vector of old image

def testModelYng_(**kwargs):
    testModelYng = testModelYng 
    return testModelYng

def testModelOld_(**kwargs):
    testModelOld = testModelOld 
    return testModelOld

def save_checkpoint(state, filename):
    pass # save the state and filename

def trainBasicStep():
    pass

def trainTransferStep():
    pass

def test_():
    acc = 0
    target = []
    target2 = []
    query_features = []
    test_features = []
    q_features = []
    g_features = []
    features_young_weight = []
    features_young_bias = []
    wt = []
    bias = []

    pass

    total_average_precision = avg_prec 
    mean_average_precision = 1/total * total_average_precision
    
    return mean_average_precision 

def adjustLearningRate():
    pass 

