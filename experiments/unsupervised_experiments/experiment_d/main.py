'''
AIFR by coupled auto-encoder network

http://chenfeixu.com/wp-content/uploads/2014/04/CAN.pdf


See experiments/experiment_d/README.md for details
'''
from experiment_d import utils

def main():

    use_cuda = True

    coupleModel = CoupleAutoModel()
    coupleModel = pass # parallelize coupleModel

    bridgeModel = bridgeModel()
    bridgeModel = pass # parallelize bridgeModel

    testModelYng = testModelYng()
    testModelYng = pass # parallelize testModelYng

    testModelOld = testModelOld()
    testModelOld = pass # parallelize testModelOld

    params = []

    for name, param in coupleModel.named_parameters():
        params.append(param)
    
    for name, param in bridgeModel.named_parameters():
        params.append(param)
    
    for name, param in testModelYng.named_parameters():
        params.append(param)
    
    for name, param in testModelOld.named_parameters():
        params.append(param)

    
    optimizer = pass # SGD optimizer with momentum = 0.9

    pass

if __name__ == '__main__':
    main()
    