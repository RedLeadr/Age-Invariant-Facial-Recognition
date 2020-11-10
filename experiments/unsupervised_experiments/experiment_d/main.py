'''
AIFR by coupled auto-encoder network

http://chenfeixu.com/wp-content/uploads/2014/04/CAN.pdf


See experiments/experiment_d/README.md for details
'''

# imports the libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms 

from experiment_d import utils, settings, data

def main():

    use_cuda = True
    device = torch.device('cuda' if use_cuda else 'cpu')

    coupleModel = coupledAutoModel()
    coupleModel = nn.DataParallel(coupleModel).to(device) # parallelize coupleModel

    bridgeModel = bridgeModel()
    bridgeModel = nn.DataParallel(bridgeModel).to(device) # parallelize bridgeModel

    testModelYng = testModelYng()
    testModelYng = nn.DataParallel(testModelYng).to(device) # parallelize testModelYng

    testModelOld = testModelOld()
    testModelOld = nn.DataParallel(testModelOld).to(device) # parallelize testModelOld

    params = []

    for name, param in coupleModel.named_parameters():
        params.append(param)
    
    for name, param in bridgeModel.named_parameters():
        params.append(param)
    
    for name, param in testModelYng.named_parameters():
        params.append(param)
    
    for name, param in testModelOld.named_parameters():
        params.append(param)

    
    optimizer = torch.optim.SGD(params, lr = settings.args.lr, momentum = 0.9) # SGD optimizer with momentum = 0.9

    train_transform = transformers.Compose([transforms.Resize(35, interpolation = 4), transforms.RandomCrop(32), transforms.ToTensor()])
    validation_transform = transforms.Compose([transforms.Resize(32, interpolation = 4), transforms.ToTensor()])

    ''' Train Loader'''
    train_loader = DataLoader(data(transform = train_transform, istrain = True, isvalid = False, isquery = False, isgall1 = False, isgall2 = False, isgall3 = False),
                    batch_size = settings.args.batch_size, shuffle = True,
                    num_workers = settings.args.workers, pin_memory = False) # data is the dataset we will use

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 5)

    '''Validation Loader'''
    valid_query_loader = DataLoader(data(transform = validation_transform, istrain = False, isvalid = True, isquery = True, isgall1 = False, isgall2 = False, isgall3 = False),
                        batch_size = settings.args.batch_size, shuffle = False,
                        num_workers = settings.args.workers, pin_memory = False) 
    valid_gall1_loader = DataLoader(data(transform = validation_transform, istrain = False, isvalid = True, isquery = True, isgall1 = True, isgall2 = False, isgall3 = False),
                        batch_size = settings.args.batch_size, shuffle = False,
                        num_workers = settings.args.workers, pin_memory = False) 
    valid_gall2_loader = DataLoader(data(transform = validation_transform, istrain = False, isvalid = True, isquery = True, isgall1 = False, isgall2 = True, isgall3 = False),
                        batch_size = settings.args.batch_size, shuffle = False,
                        num_workers = settings.args.workers, pin_memory = False) 
    valid_gall3_loader = DataLoader(data(transform = validation_transform, istrain = False, isvalid = True, isquery = True, isgall1 = False, isgall2 = False, isgall3 = True),
                        batch_size = settings.args.batch_size, shuffle = False,
                        num_workers = settings.args.workers, pin_memory = False) 

    '''Test Loader'''
    test_query_loader = DataLoader(data(transform = validation_transform, istrain = False, isvalid = False, isquery = True, isgall1 = False, isgall2 = False, isgall3 = False),
                        batch_size = settings.args.batch_size, shuffle = False,
                        num_workers = settings.args.workers, pin_memory = False)
    test_gall1_loader = DataLoader(data(transform = validation_transform, istrain = False, isvalid = False, isquery = True, isgall1 = True, isgall2 = False, isgall3 = False),
                        batch_size = settings.args.batch_size, shuffle = False,
                        num_workers = settings.args.workers, pin_memory = False)
    test_gall2_loader = DataLoader(data(transform = validation_transform, istrain = False, isvalid = False, isquery = True, isgall1 = False, isgall2 = True, isgall3 = False),
                        batch_size = settings.args.batch_size, shuffle = False,
                        num_workers = settings.args.workers, pin_memory = False)
    test_gall3_loader = DataLoader(data(transform = validation_transform, istrain = False, isvalid = False, isquery = True, isgall1 = False, isgall2 = False, isgall3 = True),
                        batch_size = settings.args.batch_size, shuffle = False,
                        num_workers = settings.args.workers, pin_memory = False)

    criterion = nn.MSELoss().to(device)

    '''Test Driver'''
    for epoch in range(settings.args.start_epoch, settings.args.epochs):

        adjust_learning_rate(optimizer, epoch)
        for param_group in optimizer.param_groups:
            print('lr after', param_group['lr'])

        '''Test One'''
        test_mean_average_prec_1 = utils.test(test_query_loader, test_gall1_loader, coupleModel, bridgeModel, testModelYng, testModelOld, device)
        print('test mean average precision for gallery1: {8f}'.format(test_mean_average_prec_1)) 

        '''Test Two'''
        test_mean_average_prec_2 = utils.test(test_query_loader, test_gall2_loader, coupleModel, bridgeModel, testModelYng, testModelOld, device)
        print('test mean average precision for gallery2: {8f}'.format(test_mean_average_prec_2)) 

        '''Test Three'''
        test_mean_average_prec_3 = utils.test(test_query_loader, test_gall3_loader, coupleModel, bridgeModel, testModelYng, testModelOld, device)
        print('test mean average precision for gallery3: {8f}'.format(test_mean_average_prec_3)) 

        # start the timer
        start = time.time()

        '''Epoch 1 loss'''
        epoch_loss_1 = utils.train_basic_step(train_loader, coupleModel, criterion, optimizer, epoch, device)
        print('\n train_basic_loss: {.6f}, Epoch: {:d} \n'.format(epoch_loss_1, epoch))

        '''Epoch 2 loss'''
        epoch_loss_2 = utils.train_basic_step(train_loader, coupleModel, criterion, optimizer, epoch, device)
        print('\n train_basic_loss: {.6f}, Epoch: {:d} \n'.format(epoch_loss_2, epoch))

        epoch_loss = epoch_loss_1 + epoch_loss_2
        
        # end the timer
        end = time.time()

        print('\n total_loss: {.6f}, Epoch: {:d} Epochtime: {:2.2f}\n'.format(epoch_loss, epoch + 1, (end - start)))


        '''Valid Mean Precision Averages of galleries 1, 2, and 3'''
        valid_mean_average_precision_1 = utils.test(valid_gall1_loader, coupleModel, bridgeModel, device)
        print('Test mean average precision for gallery 1(valid) : {.8f}'.format(valid_mean_average_precision_1))

        valid_mean_average_precision_2 = utils.test(valid_gall2_loader, coupleModel, bridgeModel, device)
        print('Test mean average precision for gallery 2(valid) : {.8f}'.format(valid_mean_average_precision_2))

        valid_mean_average_precision_3 = utils.test(valid_gall3_loader, coupleModel, bridgeModel, device)
        print('Test mean average precision for gallery 3(valid) : {.8f}'.format(valid_mean_average_precision_3))

        scheduler.step(epoch_loss)

        '''Save the checkpoint if epoch loss is less than 45'''
        save_name = settings.args.save_path + 'bridgeModel' + str(epoch) + '_checkpoint.pth.tar'
        utils.save_checkpoint({'epoch': epoch,
                    'state_dict': bridgeModel.state_dict(), 'optimizer': optimizer.state_dict()}, 
                    '1' + save_name) 

if __name__ == '__main__':
    main() 