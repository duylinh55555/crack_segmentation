import torch
from torch import nn
from unet.unet_transfer import UNet16, UNetResNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
from data_loader import ImgDataSet
import os
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
from evaluation import *
import csv
import time
import datetime

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_model(device, args):
    type ='vgg16'

    type = args.model_type

    if type == 'vgg16':
        print('create vgg16 model')
        model = UNet16(pretrained=True)
    elif type == 'resnet101':
        encoder_depth = 101
        num_classes = 1
        print('create resnet101 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    elif type == 'resnet34':
        encoder_depth = 34
        num_classes = 1
        print('create resnet34 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    elif type == 'U_Net':
        model = U_Net(img_ch=args.img_ch, output_ch=args.output_ch)
    elif type == 'R2U_Net':
        model = R2U_Net(img_ch=args.img_ch, output_ch=args.output_ch,t=args.t)
    elif type == 'AttU_Net':
        model = AttU_Net(img_ch=args.img_ch, output_ch=args.output_ch)
    elif type == 'R2AttU_Net':
        model = R2AttU_Net(img_ch=args.img_ch, output_ch=args.output_ch,t=args.t)
    else:
        assert False
    model.eval()
    return model.to(device)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def find_latest_model_path(dir):
    model_paths = []
    epochs = []
    for path in Path(dir).glob('*.pt'):
        if 'lastest_model' in path.stem:
            return path
        else:
            return None

def train(train_loader, model, criterion, optimizer, validation, args):

    latest_model_path = find_latest_model_path(args.model_dir)

    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])
        epoch = epoch

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_val_los = best_state['valid_loss']

        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        best_epoch = 0
        epoch = 0
        best_unet_score = 0.
        min_val_los = 9999

    valid_losses = []
    for epoch in range(epoch, args.n_epoch + 1):
        iter_start_time = time.time()
        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()

        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        length = 0

        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()

            masks_pred = model(input_var)

            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = target_var.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            losses.update(loss)
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc += get_accuracy(masks_pred,target_var)
            SE += get_sensitivity(masks_pred,target_var)
            SP += get_specificity(masks_pred,target_var)
            PC += get_precision(masks_pred,target_var)
            F1 += get_F1(masks_pred,target_var)
            JS += get_JS(masks_pred,target_var)
            DC += get_DC(masks_pred,target_var)
            length += input_var.size(0)

        acc = acc/length
        SE = SE/length
        SP = SP/length
        PC = PC/length
        F1 = F1/length
        JS = JS/length
        DC = DC/length
        unet_score = JS + DC

        # Print the log info
        print('\nAcc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (acc,SE,SP,PC,F1,JS,DC))
        f = open(os.path.join(args.result_dir,'train_log.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([epoch,acc,losses.avg,SE,SP,PC,F1,JS,DC,unet_score])
        f.close()

        # Validation
        valid_metrics = validation(model, valid_loader, criterion, epoch, iter_start_time)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        tq.close()

        #save the model of the current epoch
        epoch_model_path = os.path.join(*[args.model_dir, 'lastest_model.pt'])
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': losses.avg
        }, epoch_model_path)

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, best_model_path)

def validate(model, val_loader, criterion, current_epoch, iter_start_time):
    losses = AverageMeter()
    model.eval()

    acc = 0.	# Accuracy
    SE = 0.		# Sensitivity (Recall)
    SP = 0.		# Specificity
    PC = 0. 	# Precision
    F1 = 0.		# F1 Score
    JS = 0.		# Jaccard Similarity
    DC = 0.		# Dice Coefficient
    length=0

    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)
            loss = criterion(output, target_var)

            losses.update(loss.item(), input_var.size(0))

            acc += get_accuracy(input_var,target_var)
            SE += get_sensitivity(input_var,target_var)
            SP += get_specificity(input_var,target_var)
            PC += get_precision(input_var,target_var)
            F1 += get_F1(input_var,target_var)
            JS += get_JS(input_var,target_var)
            DC += get_DC(input_var,target_var)
            length += input_var.size(0)

        acc = acc/length
        SE = SE/length
        SP = SP/length
        PC = PC/length
        F1 = F1/length
        JS = JS/length
        DC = DC/length
        unet_score = JS + DC

        print(f'\tvalid_loss = {losses.avg:.5f}')
        print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, unet_score: %.4f'%(acc,SE,SP,PC,F1,JS,DC,unet_score))

        current_epoch_time = time.time() - iter_start_time
        print("Time: " + str(current_epoch_time))

        dt_string = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f = open(os.path.join(args.result_dir,'val_log.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([current_epoch,acc,losses.avg,SE,SP,PC,F1,JS,DC,unet_score,current_epoch_time,dt_string])
        f.close()

    return {'valid_loss': losses.avg}

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')

def calc_crack_pixel_weight(mask_dir):
    avg_w = 0.0
    n_files = 0
    for path in Path(mask_dir).glob('*.*'):
        n_files += 1
        m = ndimage.imread(path)
        ncrack = np.sum((m > 0)[:])
        w = float(ncrack)/(m.shape[0]*m.shape[1])
        avg_w = avg_w + (1-w)

    avg_w /= float(n_files)

    return avg_w / (1.0 - avg_w)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-n_epoch', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('-num_workers', default=4, type=int, help='output dataset directory')
    parser.add_argument('-img_ch', type=int, default=3)
    parser.add_argument('-output_ch', type=int, default=1)
    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    

    parser.add_argument('-data_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_dir', type=str, help='output dataset directory')
    parser.add_argument('-model_type', type=str, required=False, default='resnet101', choices=['vgg16', 'resnet101', 'resnet34', 'U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net'])
    parser.add_argument('-result_dir', type=str, help='results log directory')

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    DIR_TRAIN = os.path.join(args.data_dir, 'train')
    DIR_TRAIN_GT = os.path.join(args.data_dir, 'train_GT_jpg')
    DIR_VAL = os.path.join(args.data_dir, 'valid')
    DIR_VAL_GT = os.path.join(args.data_dir, 'valid_GT_jpg')
    DIR_TEST = os.path.join(args.data_dir, 'test')
    DIR_TEST_GT = os.path.join(args.data_dir, 'test_GT_jpg')

    train_img_names  = [path.name for path in Path(DIR_TRAIN).glob('*.jpg')]
    train_mask_names = [path.name for path in Path(DIR_TRAIN_GT).glob('*.jpg')]
    val_img_names  = [path.name for path in Path(DIR_VAL).glob('*.jpg')]
    val_mask_names = [path.name for path in Path(DIR_VAL_GT).glob('*.jpg')]
    test_img_names  = [path.name for path in Path(DIR_TEST).glob('*.jpg')]
    test_mask_names = [path.name for path in Path(DIR_TEST_GT).glob('*.jpg')]

    print(f'Train images = {len(train_img_names)}')
    print(f'Validate images = {len(val_img_names)}')
    print(f'Test images = {len(test_img_names)}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(device, args)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #crack_weight = 0.4*calc_crack_pixel_weight(DIR_MASK)
    #print(f'positive weight: {crack_weight}')
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([crack_weight]).to('cuda'))
    criterion = nn.BCEWithLogitsLoss().to('cuda')

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

    # dataset = ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=train_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms)
    # train_size = int(0.7*len(dataset))
    # valid_size = len(dataset) - train_size
    # train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataset   = ImgDataSet(img_dir=DIR_TRAIN, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=DIR_TRAIN_GT, mask_fnames=train_mask_names,  mask_transform=mask_tfms)
    valid_dataset   = ImgDataSet(img_dir=DIR_VAL,   img_fnames=val_img_names,   img_transform=train_tfms, mask_dir=DIR_VAL_GT,   mask_fnames=val_mask_names,    mask_transform=mask_tfms)
    test_dataset    = ImgDataSet(img_dir=DIR_TEST,  img_fnames=test_img_names,  img_transform=train_tfms, mask_dir=DIR_TEST_GT,  mask_fnames=test_mask_names,   mask_transform=mask_tfms)


    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

    model.cuda()

    train(train_loader, model, criterion, optimizer, validate, args)
