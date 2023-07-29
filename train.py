import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchsummary import summary
cudnn.benchmark = True

from models import *
from dataloaders.nyu import NYUDataset
from metrics import Metrics, AverageMetrics
from lossfunc import LossFunctions
import utils
from thop import profile


def main():

    args = utils.parse_commands('train')

    # Load data
    print('Load data...')
    traindir = os.path.join('..', 'data', args.data, 'train')
    valdir = os.path.join('..', 'data', args.data, 'val')
    train_dataset = NYUDataset(traindir, split='training', lowRes=args.lowRes, min_depth=1e-3, max_depth=10.0, num_samples=500)
    val_dataset = NYUDataset(valdir, split='validation', lowRes=args.lowRes, min_depth=1e-3, max_depth=10.0, num_samples=500)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                num_workers=args.workers, 
                                                pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=1, 
                                                shuffle=False, 
                                                num_workers=args.workers, 
                                                pin_memory=True)
    print("=> data loaders created.")
    # model_path = args.check_point

    # Load model
    # model = AutoEncoder_simpleUp(lowRes=args.lowRes).to(device)
    # model = AutoEncoder_UpCSPN(lowRes=args.lowRes, k=3).to(device)
    # model = AutoEncoder_UpCSPN(lowRes=args.lowRes, k=5).to(device)
    # model = AutoEncoder_UpCSPN(lowRes=args.lowRes, k=7).to(device)

    # model = AutoEncoder_simpleUp_bins_DR(lowRes=args.lowRes).to(device)
    # model = AutoEncoder_UpCSPN_bins_DR(lowRes=args.lowRes, k=3).to(device)
    # model = AutoEncoder_UpCSPN_bins_DR(lowRes=args.lowRes, k=5).to(device)
    # model = AutoEncoder_UpCSPN_bins_DR(lowRes=args.lowRes, k=7).to(device)

    # Bins ablation study
    # model = AutoEncoder_simpleUp_bins_DR(lowRes=args.lowRes, init_bins=1).to(device)
    # model = AutoEncoder_simpleUp_bins_DR(lowRes=args.lowRes, init_bins=2).to(device)
    # model = AutoEncoder_simpleUp_bins_DR(lowRes=args.lowRes, init_bins=4).to(device)
    # model = AutoEncoder_simpleUp_bins_DR(lowRes=args.lowRes, init_bins=8).to(device)

    # AdaBins
    model = Autoencoder_simpleUp_AdaBins(lowRes=args.lowRes).to(device)
    # model = Autoencoder_UpCSPN_AdaBins(lowRes=args.lowRes, k=3).to(device)
    # model = Autoencoder_UpCSPN_AdaBins(lowRes=args.lowRes, k=5).to(device)
    # model = Autoencoder_UpCSPN_AdaBins(lowRes=args.lowRes, k=7).to(device)


    in_size1 = torch.randn(1, 3, 224, 224).to(device)
    in_size2 = torch.randn(1, 1, 224, 224).to(device)
    macs, params = profile(model,
                            inputs=(in_size1, in_size2),
                            verbose=False)

    print('MACs = {}G'.format(macs/1e9))
    print('Params = {}M'.format(params/1e6))

    # Parameters
    criterion = LossFunctions()
    if args.is_bin == 1:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.95, 0.99), weight_decay=0.1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.8)

    # Train Progress
    train(train_loader, val_loader, model, criterion, optimizer, scheduler, device, args)


def train(train_loader, val_loader, model, criterion, optimizer, scheduler, device, args):
    # print(model)
    print('Training Start!, batch_size = {}, epochs = {}, workers = {}, bins = {}'.format(args.batch_size, args.epochs, args.workers, args.bins*8))
    print('Model: {}'.format(model.name))
    print('optimizer: {}'.format(optimizer))
    threshold = 100.0

    t_iteration = 0
    v_iteration = 0

    # Loss functions
    criterion_si_85= criterion.scale_invariant_loss(scale=0.85)
    criterion_bin = criterion.bin_chamfer_loss()
    criterion_l1 = criterion.l1_loss()
    criterion_l2 = criterion.l2_loss()


    writer = SummaryWriter('./runs/{}'.format(model.name + str(model.lowRes)))
    save_dir = './result/{}'.format(model.name) + '_lowRes_' + str(model.lowRes)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for epochs in range(args.epochs):
        print('Epoch {}: lr = {}'.format(epochs+1, scheduler.get_last_lr()))
        average_meter = AverageMetrics()

        # Train --------------------------------------------------------------------------------------------------------------------|
        model.train()
        with tqdm(train_loader, unit='batch', bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}', ncols=130) as tepoch:
            for inputs in tepoch:
                tepoch.set_description('Train')

                # Inputs
                input_rgb = inputs['rgb']
                input_depth = inputs['depth']
                target = inputs['gt']
                input_rgb, input_depth, target = input_rgb.to(device), input_depth.to(device), target.to(device)

                optimizer.zero_grad()
                torch.cuda.synchronize()
                time_start = time.time()
                if args.lowRes == 1:
                    if args.is_bin == 1:
                        bin_edges, outputs = model(input_rgb, input_depth)
                        torch.cuda.synchronize()
                        time_end = time.time()
                        loss_bin = criterion_bin(bin_edges, target)
                        loss_pixel = criterion_si_85(outputs, target)
                    else:
                        outputs, features = model(input_rgb, input_depth)
                        torch.cuda.synchronize()
                        time_end = time.time()
                        loss_bin = 0
                        loss_pixel = criterion_l1(outputs, target)
                else:
                    # Only train bin model for sparse case
                    bin_edges, outputs = model(input_rgb, input_depth)
                    torch.cuda.synchronize()
                    time_end = time.time()
                    loss_bin = criterion_bin(bin_edges, target)
                    loss_pixel = criterion_si_85(outputs, target)

                datatime = time_start - time_end
                # Total loss backward
                loss = args.pixel_weight*loss_pixel + args.bin_weight*loss_bin
                loss.backward()
                optimizer.step()

                # Evaluation
                metrics = Metrics()
                metrics.evaluate(outputs, target, loss.item(), datatime)
                average_meter.update(metrics)
                average = average_meter.average()

                tepoch.set_postfix(
                        loss = '{:.4f}'.format(average.loss),
                        delta1 = '{:.2f}'.format(100*average.delta1), 
                        RMSE = '{:.3f}'.format(average.rmse), 
                        )

                writer.add_scalar('Train/Delta1', 100.*metrics.delta1, t_iteration)
                writer.add_scalar('Train/Loss', metrics.loss, t_iteration)
                writer.add_scalar('Train/RMSE', metrics.rmse, t_iteration)
                writer.add_scalar('Train/Loss_si', loss_pixel, t_iteration)
                writer.add_scalar('Train/Loss_bin', loss_bin, t_iteration)

                output_norm = outputs/10
                target_norm = target/10

                img = torch.cat([output_norm[0, :, :, :], target_norm[0, :, :, :]], dim=2)
                writer.add_images('Train/Result', img, t_iteration, 0, dataformats='CHW')
                t_iteration += 1
                # if args.is_bin == 1:
                #     scheduler.step()
            # if args.is_bin == 0:
            scheduler.step()

        # Validation ---------------------------------------------------------------------------------------------------------------|
        v_average_meter = AverageMetrics()
        model.eval()
        with tqdm(val_loader, unit='batch', bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}', ncols=130) as tepoch:
            for (inputs) in tepoch:
                tepoch.set_description('Valid')

                # Input
                input_rgb = inputs['rgb']
                input_depth = inputs['depth']
                target = inputs['gt']
                input_rgb, input_depth, target = input_rgb.to(device), input_depth.to(device), target.to(device)
                with torch.no_grad():
                    torch.cuda.synchronize()
                    time_start = time.time()
                    if args.lowRes == 1:
                        if args.is_bin == 1:
                            bin_edges, outputs = model(input_rgb, input_depth)
                            torch.cuda.synchronize()
                            time_end = time.time()
                            loss_bin = criterion_bin(bin_edges, target)
                            loss_pixel = criterion_si_85(outputs, target)
                        else:
                            outputs, features = model(input_rgb, input_depth)
                            torch.cuda.synchronize()
                            time_end = time.time()
                            loss_bin = 0
                            loss_pixel = criterion_l1(outputs, target)
                    else:
                        # Only train bin model for sparse case
                        bin_edges, outputs = model(input_rgb, input_depth)
                        # print(outputs.size())
                        # print(target.size())
                        torch.cuda.synchronize()
                        time_end = time.time()
                        loss_bin = criterion_bin(bin_edges, target)
                        loss_pixel = criterion_si_85(outputs, target)

                    # Loss functions calculation
                    loss = args.pixel_weight*loss_pixel + args.bin_weight*loss_bin

                datatime = time_end - time_start
                v_metrics = Metrics()
                v_metrics.evaluate(outputs, target, loss.item(), datatime)
                v_average_meter.update(v_metrics)
                v_average = v_average_meter.average()

                tepoch.set_postfix(
                        loss='{:.4f}'.format(v_average.loss),
                        delta1='{:.2f}'.format(100.*v_average.delta1), 
                        RMSE='{:.3f}'.format(v_average.rmse), 
                        REL='{:.3f}'.format(v_average.absrel)
                        )

                writer.add_scalar('Validation/Delta1', 100.*v_metrics.delta1, v_iteration)
                writer.add_scalar('Validation/Loss', v_metrics.loss, v_iteration)
                writer.add_scalar('Validation/RMSE', v_metrics.rmse, v_iteration)
                writer.add_scalar('Validation/REL', v_metrics.absrel, v_iteration)

                output_norm = outputs/10
                target_norm = target/10

                img = torch.cat([output_norm[0, :, :, :], target_norm[0, :, :, :]], dim=2)
                writer.add_images('Valid/Result', img, t_iteration, 0, dataformats='CHW')
                v_iteration += 1

        
        bound = v_average.rmse + 100.*(1 - v_average.delta1)
        if(bound <= threshold):
            threshold = bound
            print('Saving model to '+ save_dir + '/ep{}_{:.2f}_rmse{:.3f}'.format(epochs+1, 100.*v_average.delta1, v_average.rmse))
            torch.save(model, save_dir + '/ep{}_{:.2f}_rmse{:.3f}.pth'.format(epochs+1, 100.*v_average.delta1, v_average.rmse))
            print('Evaluation results:')
            print('\tloss={:.4f}'.format(v_average.loss))
            print('\tdelta1={:.2f}'.format(100.*v_average.delta1))
            print('\tdelta2={:.2f}'.format(100.*v_average.delta2))
            print('\tdelta3={:.2f}'.format(100.*v_average.delta3))
            print('\tRMSE={:.3f}'.format(v_average.rmse))
            print('\tREL={:.3f}'.format(v_average.absrel))
            print('\tMax FPS: {:.2f}'.format(v_average.max_fps))
            print('\tAVG FPS: {:.2f}'.format(v_average.avg_fps))
        elif(epochs==args.epochs-1):
            print('Saving model to '+ save_dir + '/ep{}_{:.2f}_rmse{:.3f}'.format(epochs+1, 100.*v_average.delta1, v_average.rmse))
            # torch.save(model, './result/test/ep{}_{:.2f}_rmse{:.3f}.pth'.format(epochs+1, 100.*v_average.delta1, v_average.rmse))
            torch.save(model, save_dir + '/ep{}_{:.2f}_rmse{:.3f}.pth'.format(epochs+1, 100.*v_average.delta1, v_average.rmse))
            print('Evaluation results:')
            print('\tloss={:.4f}'.format(v_average.loss))
            print('\tdelta1={:.2f}'.format(100.*v_average.delta1))
            print('\tdelta2={:.2f}'.format(100.*v_average.delta2))
            print('\tdelta3={:.2f}'.format(100.*v_average.delta3))
            print('\tRMSE={:.3f}'.format(v_average.rmse))
            print('\tREL={:.3f}'.format(v_average.absrel))
            print('\tMax FPS: {:.2f}'.format(v_average.max_fps))
            print('\tAVG FPS: {:.2f}'.format(v_average.avg_fps))

if __name__=='__main__':
    torch.cuda.manual_seed(0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)
    main()
    print('\nFinished!')
