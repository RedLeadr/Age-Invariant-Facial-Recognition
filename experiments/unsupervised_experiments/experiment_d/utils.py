'''
AIFR by coupled auto-encoder network

http://chenfeixu.com/wp-content/uploads/2014/04/CAN.pdf


See experiments/experiment_d/README.md for details
'''

# imports the libraries 
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.nn.init as init 
import time 

from experiment_d import main, data, settings 

# defines the classes

class coupledAutoModel(nn.Module):
    def __init__(self):
        super(coupledAutoModel, self).__init__()

        # young architecture

        '''Encoder'''
        self.linear_encode_yng = nn.Linear(32*32, 4000)
        self.sigmoid_yng = nn.Sigmoid()

        '''Decoder'''
        self.linear_decode_yng = nn.Linear(4000, 1024)

        # old architecture
        
        '''Encoder'''
        self.linear_encode_old = nn.Linear(32*32, 4000)
        self.sigmoid_old = nn.Sigmoid()

        '''Decoder'''
        self.linear_decode_old = nn.Linear(4000, 1024)

    def initWeights(self):
        # w1
        init.normal_(self.linear_encode_yng.weight, 0, 1e-4)
        # b1
        init.constant_(self.linear_encode_yng.bias, 0)
        # w1_bar
        init.normal_(self.linear_decode_yng.weight, 0, 1e-4)
        # c1
        init.constant_(self.linear_decode_yng.bias, 0)
        
        # w2
        init.normal_(self.linear_encode_old.weight, 0, 1e-4)
        # b2
        init.constant_(self.linear_encode_old.bias, 0)
        # w2_bar
        init.normal_(self.linear_decode_old.weight, 0, 1e-4)
        # c2
        init.constant_(self.linear_decode_old.bias, 0)

    def forward(self, x1, x2):
        x1 = self.linear_encode_yng(x1.view(-1, 32 * 32)) # encode young x1
        x1 = self.sigmoid_yng(x1) # sigmoid young x1
        x1_bar = self.linear_decode_yng(x1) # decode young x1_bar

        x2 = self.linear_encode_old(x2.view(-1, 32 * 32)) # endode old x2
        x2 = self.sigmoid_old(x2) # sigmoid old x2
        x2_bar = self.linear_decode(x2) # decode old x2_bar

        c1 = self.linear_decode_yng.bias # compute c1 young bias
        c2 = self.linear_decode_old.bias # compute c2 old bias

        return x1_bar, x2_bar, c1, c2

def coupledAutoModel_(**kwargs):
    model_auto_encoder = coupledAutoModel
    return model_auto_encoder

class bridgeModel(nn.Module):
    def __init__(self):
        super(bridgeModel, self).__init__()

        '''Age Part'''
        # x1, both wv1 and bv1 are automatically calculated
        self.linear_encode_x1 = nn.Linear(32 * 32, 800)
        self.sigmoid_x1 = nn.Sigmoid()

        # after applying this to x1, you get A1(i)
        # used for x2, wv2 and bv2 are automatically calculated
        self.linear_encode_x2 = nn.Linear(32 * 32, 800)
        self.sigmoid_x2 = nn.Sigmoid()

        # no of neurons is 800
        # aging bridge network from A1 to A2 hat
        self.linear_bridge_aging_1 = nn.Linear(800, 800)
        self.sigmoid_aging_1 = nn.Sigmoid()
        self.linear_bridge_aging_2 = nn.Linear(800, 800)
        self.sigmoid_aging_2 = nn.Sigmoid()

        # deaging bridge network from A2 to A1 hat
        self.linear_bridge_deaging_1 = nn.Linear(800, 800)
        self.sigmoid_deaging_1 = nn.Sigmoid()
        self.linear_bridge_deaging_2 = nn.Linear(800, 800)
        self.sigmoid_deaging_2 = nn.Sigmoid()

        '''Identity Part'''
        self.linear_identity_yng = nn.Linear(32 * 32, 2800)
        self.sigmoid_identity_yng = nn.Sigmoid()
        self.linear_identity_old = nn.Linear(32 * 32, 2800)
        self.sigmoid_identity_old = nn.Sigmoid()

        '''Reconstructs x1_bar and x2_bar'''
        # age_young
        self.reconstructed_x1_age = nn.Linear(800, 1024)
        # identity_yng
        self.reconstructed_x1_identity = nn.Linear(2800, 1024)
        # age_old
        self.reconstructed_x2_age = nn.Linear(800, 1024)
        # identity_old
        self.reconstructed_x2_identity = nn.Linear(2800, 1024)
        self.sigmoid_recons = nn.Sigmoid()

        self.initWeights()

    def initWeights(self):
        # Ha1
        init.normal_(self.linear_bridge_aging_1.weight, 0, 1e-4)

        # Ha2
        init.normal_(self.linear_bridge_aging_1.bias, 0)

        # ba1
        init.constant_(self.linear_bridge_aging_1.bias, 0)

        # ba2
        init.constant_(self.linear_bridge_aging_2.bias, 0)

        # Hd1
        init.normal_(self.linear_bridge_deaging_1.weight, 0, 1e-4)

        # Hd2
        init.normal_(self.linear_bridge_deaging_2.weight, 0, 1e-4)

        # bd1
        init.constant_(self.linear_bridge_deaging_1.bias, 0)

        # bd2
        init.constant_(self.linear_bridge_deaging_2.bias, 0)

    def forward(self, x1, x2, c1, c2): # substitue all self.pass
        '''Age Part'''
        # aging part, goes from x1 to a1 to a2_hat
        x1_base = x1
        x1 = self.linear_encode_x1(x1) # encode x1
        a1 = self.sigmoid_x1(x1) # sigmoid of x1, gives a1(i)

        by1 = self.linear_bridge_aging_1(a1) # bridge young 1
        by1 = self.sigmoid_aging_1(by1) # sigmoid of by1
        by2 = self.linear_bridge_aging_2(by1) # bridge young 2
        a2_hat = self.sigmoid_aging_2(by2) # sigmoid of by2

        # deaging part, goes from x2 to a2 to a1_hat
        x2_base = x2
        x2 = self.linear_encode_x2(x2) # encode x2
        a2 = self.sigmoid_x2(x2) # sigmoid of x2, gives a2(i)

        bo1 = self.linear_bridge_aging_2(a2)  # bridge old 1
        bo1 = self.sigmoid_aging_1(bo1) # sigmoid of bo1
        bo2 = self.linear_bridge_aging_2(bo1) # bridge old 2
        a1_hat = self.sigmoid_aging_2(bo2) # sigmoid of bo2

        '''Identity Part'''
        id1 = self.linear_identity_yng(x1_base) # x1_base linear identity 
        id1 = self.sigmoid_identity_yng(id1) # sigmoid of id1 
        id2 = self.linear_identity_yng(x2_base) # x2_base linear identity 
        id2 = self.sigmoid_identity_yng(id2) # sigmoid of id2

        wu1 = self.linear_identity_yng.weight # linear identity young weight 
        wu2 = self.linear_identity_old.weight # linear identity old weight 
        bu1 = self.linear_identity_yng.bias # linear identity young bias 
        bu2 = self.linear_identity_old.bias # linear identity old bias 

        wv1 = self.linear_encode_x1.weight # endode x1.weight 
        wv2 = self.linear_encode_x2.weight # endode x2.weight

        '''Addition Part'''
        wv1_hat = self.reconstructed_x1_age.weight # reconstructed x1_age.weight
        wu1_hat = self.reconstructed_x1_identity.weight # reconstructed x1_identity.weight
        wv2_hat = self.reconstructed_x2_age.weight # reconstructed x2_age.weight
        wu2_hat = self.reconstructed_x2_identity.weight # reconstructed x2_identity.weight

        mul1_x1 = torch.matmul(wv1_hat, a1_hat.t()) # matmul(wv1_hat, a1_hat)
        mul2_x1 = torch.matmul(wv1_hat, id1.t()) # matmul(wu1_hat, id1)

        mul1_x1 = mul1_x1.t() # mul1_x1()
        mul2_x1 = mul2_x1.t() # mul2_x1()

        add_mul1 = torch.add(mul1_x1, mul2_x1) # add layer (mul1_x1, 1, mul2_x1)
        add_x1 = torch.add(add_mul1, 1, c1) # add layer (add_mul1, 1, c1)

        reconstruct_x1 = self.sigmoid_recons(add_x1) # sigmoid of add_x1
        mul1_x2 = torch.matmul(wv2_hat, a2_hat.t()) # matmul(wv2_hat, a2_hat())
        mul2_x2 = torch.matmul(wu2_hat, id2.t()) # matmul(wu2_hat, id2())

        mul1_x2 = mul1_x2.t() # mul1_x2()
        mul2_x2 = mul2_x2.t() # mul2_x2()

        add_mul2 = torch.add(mul1_x2, 1, mul2_x2) # add layer (mul1_x2, 1, mul2_x2)
        add_x2 = torch.add(add_mul2, 1, c2) # add layer (add_mul2, 1, c2)

        reconstruct_x2 = self.sigmoid_recons(add_x2) # sigmoid of add_x2

        return a1, a1_hat, a2, a2_hat, id1, id2, x1_base, reconstruct_x1, x2_base, reconstruct_x2, wu1, wu2, wv1, wv2, bu1, bu2
    
def bridgeModel_(**kwargs):
    bridgeModel = bridgeModel 
    return bridgeModel
    
class testModelYng(nn.Module):
    def __init__(self):
        super(testModelYng, self).__init__()
        self.linear_encode_yng = nn.Linear(1024, 2800)
        self.sigmoid = nn.Sigmoid()

        # pre_bridge_dict = torch.load() 
        # self.pre_trained_dict = pre_bridge_dict

        self.initWeights()
    
    def initWeights(self):
        self.linear_encode_yng.weight = nn.Parameter(self.pre_trained_dict['module.linear_identity_yng.weight'])
        self.linear_encode_yng.bias = nn.Parameter(self.pre_trained_dict['module.linear_identity_yng.bais'])

    def forward(self, x1):
        x1 = self.linear_encode_yng(x1) # encode of x1
        x1 = self.sigmoid(x1) # sigmoid of x1

        features_yng_weight = self.linear_encode_yng.weight # linear encode of yng.weight
        features_yng_bias = self.linear_encode_yng.bias # linear encode of yng.bias

        return x1, features_yng_weight, features_yng_bias # feature vector of young image

class testModelOld(nn.Module):
    def __init__(self):
        super(testModelOld, self).__init__()
        self.linear_encode_old = nn.Linear(1024, 2800)
        self.sigmoid = nn.Sigmoid()

        # pre_bridge_dict = torch.load()
        # self.pre_trained_dict = pre_bridge_dict['state_dict']
        self.initWeights()
    
    def initWeights(self):
        self.linear_encode_old.weight = nn.Parameter(self.pre_trained_dict['module.linear_identity_old.weight'])
        self.linear_encode_old.bias = nn.Parameter(self.pre_trained_dict['module.linear_identity_old.bias'])

    def forward(self, x2):
        x2 = self.linear_encode_old(x2) # encode of x2
        x2 = self.sigmoid(x2) # sigmoid of x2

        features_old_weight = self.linear_encode_old.weight # linear encode of old.weights
        features_old_bias = self.linear_encode_old.bias # linear encode of old.bias

        return x2, features_old_weight, features_old_bias # feature vector of old image

def testModelYng_(**kwargs):
    testModelYng = testModelYng 
    return testModelYng

def testModelOld_(**kwargs):
    testModelOld = testModelOld 
    return testModelOld

def save_checkpoint(state, filename):
    torch.save(state, filename) # save the state and filename

def trainBasicStep(train_loader, coupledAutoModel, criterion, optimizer, epoch, device):
    running_loss = 0
    data_size = 0

    for (label, x1, x2) in train_loader: 
        optimizer.zero_grad()
        x1 = x1.to(device)
        x2 = x2.to(device)
        x1 = x1.view(-1, 32 * 32)
        x2 = x2.view(-1, 32 * 32)
        label = torch.tensor(label, dtype = torch.long)
        label = label.to(device)
        recons_x1, recons_x2, __, __ = coupledAutoModel(x1, x2)
        loss_x1 = criterion(x1, recons_x1)
        loss_x2 = criterion(x2, recons_x2)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * label.size(0)
        data_size += label.size(0)
    return running_loss / data_size

def trainTransferStep(train_loader, coupledAutoModel, bridgeModel, criterion, optimizer, epoch, device):
    running_loss = 0
    data_size = 0
    for (label, x1, x2) in train_loader:
        optimizer.zero_grad()
        x1 = x1.to(device)
        x2 = x2.to(device)
        x1 = x1.view(-1, 32 * 32)
        x2 = x2.view(-1, 32 * 32)
        label = torch.tensor(label, dtype = torch.long)
        label = label.to(device)
        __, __, c1, c2 = coupledAutoModel(x1, x2)
        a1, a1_hat, a2, a2_hat, id1, id2, x1, recons_x1, x2, x2_recons, wu1, wu2 = bridgeModel(x1, x2, c1, c2)
        loss_a1 = criterion(a1, a1_hat)
        loss_a2 = criterion(a2, a2_hat)
        loss_id = criterion(id1, id2)
        loss_x1 = criterion(x1, recons_x1)
        loss_x2 = criterion(x2, recons_x2)
        total_loss = loss_a1 + loss_a2 + loss_id + loss_x1 + loss_x2
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item() * label.size(0)
        data_size += label.size(0)
    
    return running_loss / data_size

def test_(test_query_loader, test_gall_loader, coupledAutoModel, bridgeModel, testModelYng, testModelOld, device):
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

    with torch.no_grad():
        for (x1, age1, label1) in test_query_loader: 
            x1 = x1.to(device)
            x1 = x1.view(-1, 32 * 32)

           pass

    

    total_average_precision = avg_prec 
    mean_average_precision = 1/total * total_average_precision
    
    return mean_average_precision 

def adjustLearningRate():
    for param_group in optimizer.param_groups:
        if epoch > 3:
            param_group['lr'] = 0.0001