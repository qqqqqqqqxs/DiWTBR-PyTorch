import argparse
import os
import torch
import util
import skimage.color as sc
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.DiWTBR_S import DiWTBR
from model.DiWTBR_T import DiWTBR_T
import piq
from model.data_load import TestData, DefocusData
from model.my_loss import VGGPerceptualLoss
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler
import lpips
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description = 'DiWTBR')
parser.add_argument("--model_type", type=str, default='DiWTBR-B', 
                    choices=['DiWTBR-B', 'DiWTBR-S', 'DiWTBR-T'], 
                    help="Model architecture")
parser.add_argument("--trainset_input", type=str, default="./input/F16", help="Training set input images path")
parser.add_argument("--trainset_output", type=str, default="./input/F2", help="Training set output images path")
parser.add_argument("--validset_input", type=str, default="./validation/F16", help="Testing set input images path")
parser.add_argument("--validset_output", type=str, default="./validation/F2", help="Testing set output images path")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=300, help="number of epochs to train for")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
# parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default=r"", type=str, help="Path to checkpoint and optimizer (default: none)")
parser.add_argument("--CKPT_dir", default=r"./Checkpoint/B_EBB_resume.pth", type=str, help="Checkpoint storage path (default: none)")
parser.add_argument("--weights_only", default=r"./Checkpoint/weightsonly/B_EBB.pth", type=str,help = 'Storage of only the weights of the Checkpoint')
parser.add_argument("--Tensorboard_dir", default=r"./Tensorboard/DiWTBR", type=str,help = 'Storage of only the weights of the checkpoint')
parser.add_argument("--isY", action="store_true", default=False,help = 'YCbcr use')

def data_normal(origin_data):
    dmin = origin_data.min()
    if dmin < 0:
        origin_data += torch.abs(dmin)
        dmin = origin_data.min()
    dmax = origin_data.max()
    dst = dmax - dmin
    norm_data = (origin_data - dmin).true_divide(dst)
    return norm_data



def train(training_data_loader, optimizer, lr_scheduler, model, criterion1, criterion2, criterion3, epoch, writer):

    # lr = adjust_learning_rate(optimizer, epoch - 1)
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr 
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    num_steps = len(training_data_loader)
    loss_out = 0
    loss_ssim = 0
    loss_perceptual = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        optimizer.zero_grad()
        image,label = Variable(batch[0], requires_grad=False),Variable(batch[1], requires_grad=False)
        image = image.cuda()
        label1 = label.cuda()
        defocus1 = model(image)
        loss_mse1 = criterion1(defocus1, label1)
        loss2 = piq.SSIMLoss(data_range=1.)(torch.clamp(defocus1, 0.0, 1.0), label1)
        loss3 = criterion3(defocus1, label1)
        loss_ssim = loss_ssim + loss2.data.item()
        loss_perceptual = loss_perceptual + loss3.data.item()
        loss = 0.1*loss2 + 1e-3 * loss3 + 2*loss_mse1# + 0.6*loss_mse2 + 0.4*loss_mse3 + 0.2*loss_mse4
        loss_out = loss_out + loss.data.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + iteration)
    print("===> Epoch[{}]: Loss: {:.10f}".format(epoch, loss_out/len(training_data_loader)))
    writer.add_scalar('allloss', loss_out/len(training_data_loader), epoch)
    writer.add_scalar('loss_ssim', 0.1*loss_ssim / len(training_data_loader), epoch)
    writer.add_scalar('loss_perceptual', 1e-3*loss_perceptual / len(training_data_loader), epoch)

def save_last_checkpoint(epoch,model,optimizer):
    torch.save({
            #'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 
            opt.CKPT_dir)

def validation(model):
    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])
    test_set = TestData(opt.validset_input, opt.validset_output, transform1,  data_agu=False, is_training=False)
    test_data_loader = DataLoader(dataset=test_set, num_workers=3, batch_size=1, shuffle=False)
    #model.eval()
    with torch.no_grad():
        psnr,ssim,lpips_sum = 0,0,0
        avg_psnr,avg_ssim,avg_lpips =0,0,0
        lossfn = lpips.LPIPS(net= "alex")
        for iteration, batch in enumerate(test_data_loader, 1):
            image,label = batch[0],batch[1]
            label_h, label_w = batch[2][0],batch[3][0]
            defocus = model(image)
            defocus = F.interpolate(defocus, size=(label_h, label_w), mode='bilinear', align_corners=True)
            defocus = defocus.cpu()
            defocus = defocus.squeeze().clamp(0, 1)
            label = label.squeeze().clamp(0, 1)
            lpips_dis = lossfn(defocus, label)
            defocus = defocus.detach().numpy() * 255
            defocus = defocus.astype(np.uint8)
            defocus = np.moveaxis(defocus, 0, -1)
            # label = F.interpolate(label, size=(label_h, label_w), mode='bilinear', align_corners=True)
            label = label.detach().numpy() * 255
            label = label.astype(np.uint8)
            label = np.moveaxis(label, 0, -1)
            if opt.isY is True:
                defocus_Y = util.quantize(sc.rgb2ycbcr(defocus)[:, :, 0])
                label_Y =  util.quantize(sc.rgb2ycbcr(label)[:, :, 0])
            else:
                defocus_Y = defocus
                label_Y = label
            lpips_sum += lpips_dis
            psnr += util.compute_psnr(defocus_Y, label_Y)
            ssim += util.compute_ssim(defocus_Y, label_Y)
            #psnr = util.compute_psnr(defocus_Y, label_Y)
            #print(psnr)

        avg_psnr = psnr/len(test_data_loader)
        avg_ssim = ssim/len(test_data_loader)
        avg_lpips = lpips_sum/len(test_data_loader)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f},lpips: {:.4f}".format(avg_psnr, avg_ssim, avg_lpips.item()))
    return avg_psnr,avg_ssim

if __name__ == "__main__":
    global opt, model
    opt = parser.parse_args(args=[])
    print(opt)
    
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    
    linear_scaled_lr = 3e-4  #2e-4 * 2
    linear_scaled_warmup_lr = 2e-7
    linear_scaled_min_lr = 2e-7 * 2

    print("===> Building model")
    if opt.model_type == 'DiWTBR-B':
        model = DiWTBR(basic_ch=64)
    elif opt.model_type == 'DiWTBR-S':
        model = DiWTBR(basic_ch=32)  
    elif opt.model_type == 'DiWTBR-T':
        model = DiWTBR_T(basic_ch=32)  
    else:
        raise ValueError(f"Unsupported model type: {opt.model_type}")
    model = nn.DataParallel(model)
    model = model.cuda()
    criterion1 = nn.MSELoss().cuda()
    criterion2 = None
    criterion3 = VGGPerceptualLoss(resize=False).cuda()
    
    optimizer = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),eps=1e-8, betas=(0.9, 0.999),lr=linear_scaled_lr, weight_decay=0.05)
    
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            #opt.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)  
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            validation(model)
            print('test end')
        else:
            print("=> no model found at '{}'".format(opt.resume))
    print("===> Setting Optimizer")
    
    print("===> Loading  training datasets")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = DefocusData(opt.trainset_input, opt.trainset_output, transform=transform)
    training_data_loader = DataLoader(dataset=train_set, 
                                      num_workers=8, 
                                      batch_size=opt.batchSize, 
                                      shuffle=True, 
                                      drop_last=True)
    n_iter_per_epoch = len(training_data_loader)
    num_steps = opt.nEpochs * len(training_data_loader)
    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=linear_scaled_min_lr,
            warmup_lr_init=linear_scaled_warmup_lr,
            warmup_t=20 * n_iter_per_epoch,
            cycle_limit=1,
            t_in_epochs=False,
        )

    print("===> Training")

    psnr = 0
    avgssim = 0
    writer = SummaryWriter(opt.Tensorboard_dir)
    current_time = datetime.datetime.now()
    print(current_time)


    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        # train(training_data_loader, optimizer, lr_scheduler, model, criterion1, criterion2, criterion3, epoch, writer)
        # psnr,avgssim = validation(model)
        if epoch < 10:
            psnr,avgssim = validation(model)
        if epoch > 200:
            psnr,avgssim = validation(model)
        
        if opt.CKPT_dir:
            save_last_checkpoint(epoch,model,optimizer)
        writer.add_scalar('psnr', psnr, epoch)

    current_time = datetime.datetime.now()
    print(current_time)

    if opt.weights_only:
        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
            opt.weights_only)
    