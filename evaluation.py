import os
from tqdm import tqdm
# from ptflops import get_model_complexity_info
import time
import argparse
# import matplotlib as mpl
# mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
cudnn.benchmark = True

from dataloaders.nyu import NYUDataset
from metrics import Metrics, AverageMetrics
import lossfunc
import utils

# from models.models import *

# im_height, im_width = 224, 224
im_height, im_width = 256, 320

def main():
    args = utils.parse_commands('eval')
   
    # Load data
    print('Load data...')
    valdir = os.path.join('..', 'data', 'nyudepthv2', 'val')
    val_dataset = NYUDataset(valdir, split='validation', lowRes=args.lowRes, min_depth=1e-3, max_depth=np.inf)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=1, 
                                                shuffle=False, 
                                                num_workers=24, 
                                                pin_memory=True)
    print("=> data loaders created.")

    model = torch.load(args.weight, map_location=device)

    criterion = lossfunc.LossFunctions()

    if args.mode == 'e':
        evaluation(val_loader, model, criterion, device, args)
    elif args.mode == 'v':
        visualize(val_loader, model, criterion, device, args)
    else:
        raise(RuntimeError('Mode is not defined.'))
        

# For visualization results
def visualize(val_loader, model, criterion, device, args):
    print('Visualization start!')
    print('Model: {}'.format(model.name))
    average_meter = AverageMetrics()

    model.eval()
    print(model.name)
    index = 0
    save_dir = './imgs/{}'.format(model.name + '_lowRes_' + str(model.lowRes))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with tqdm(val_loader, unit='batch', bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as tepoch:
        for inputs in tepoch:
            input_rgb = inputs['rgb']
            input_depth = inputs['depth']
            target = inputs['gt']
            input_rgb, input_depth, target = input_rgb.to(device), input_depth.to(device), target.to(device)
            with torch.no_grad():
                if args.is_bin == 0:
                    outputs, features = model(input_rgb, input_depth)
                else:
                    bin_edges, outputs = model(input_rgb, input_depth)

            if args.lowRes == 1:
                input_depth = F.interpolate(input_depth, (224, 224), mode='nearest')

            input_rgb_np = input_rgb.cpu().detach().numpy()
            input_depth_np = input_depth.cpu().detach().numpy()
            output_np = outputs.cpu().detach().numpy()

            gt_np = target.cpu().detach().numpy()
            gt_flatten_np = np.reshape(gt_np, im_width*im_height)
            # print(gt_flatten_np.shape)
            # print(bin_edges.size())
            if args.is_bin == 1:
                B_edges = bin_edges[3]
                B_centers = 0.5 * (B_edges[:, :-1] + B_edges[:,1:])
                B_centers = B_centers.squeeze(0)
                B_centers_np = B_centers.cpu().detach().numpy()
                y = np.ones_like(B_centers_np)*-10

                fig, ax = plt.subplots()

                ax.hist(gt_flatten_np, bins=1024, linewidth=0.5, label='GT dist.')
                ax.plot(B_centers_np, y, '|', ms=20, label='Bin centers')
                ax.legend(fontsize='x-large')
                # ax.grid(True, which='both')
                plt.axhline(0, color='black', linewidth=.5)
                plt.axvline(0, color='black', linewidth=.5)
                # print(B_centers_np)

                ax.set(xlim=(0, 10), xticks=np.arange(1, 10), xlabel='Depth range', 
                        ylim=(-10, 300), ylabel='number of depth values')
                # plt.show()
                plt.savefig(save_dir + '/{0:04}_bins.png'.format(index))



            input_depth_np = (input_depth_np/10)*255.
            output_np = (output_np/10)*255.
            gt_np = (gt_np/10)*255.
            input_rgb_np = np.transpose(np.reshape(input_rgb_np, (3, im_height, im_width)), (1, 2, 0))
            input_rgb_np = cv2.cvtColor(input_rgb_np*255., cv2.COLOR_RGB2BGR)

            input_depth_np = np.transpose(np.reshape(input_depth_np, (1, input_depth_np.shape[2], input_depth_np.shape[3])), (1, 2, 0))
            output_np = np.transpose(np.reshape(output_np, (1, im_height, im_width)), (1, 2, 0))
            gt_np = np.transpose(np.reshape(gt_np, (1, im_height, im_width)), (1, 2, 0))

            c1 = output_np.shape[0] // 2
            c2 = output_np.shape[1] // 2
            x = 8
            y = 14
            cropx = im_width - 2*x
            cropy = im_height - 2*y

            # x = c1 - im_height//2 - 8
            # y = c2 - im_width//2 - 14

            # rgb_np = np.zeros((cropy, cropx, 3))
            # for i in range(3):
                # rgb_np[:, :, i] = input_rgb_np[int(y):int(y+cropy), int(x):int(x+cropx), i]

            input_depth_np = cv2.applyColorMap(input_depth_np.astype(np.uint8), 15)
            output_np = cv2.applyColorMap(output_np.astype(np.uint8), 15)
            gt_np = cv2.applyColorMap(gt_np.astype(np.uint8), 15)

            # input_depth_np = input_depth_np[int(y):int(y+cropy), int(x):int(x+cropx)]
            # # print(input_depth_np)
            # output_np = output_np[int(y):int(y+cropy), int(x):int(x+cropx)]
            # gt_np = gt_np[int(y):int(y+cropy), int(x):int(x+cropx)]

            cv2.imwrite(save_dir + '/{0:04}_rgb.png'.format(index), input_rgb_np) 
            cv2.imwrite(save_dir + '/{0:04}_sd.png'.format(index), input_depth_np) 
            cv2.imwrite(save_dir + '/{0:04}_depth.png'.format(index), output_np) 
            cv2.imwrite(save_dir + '/{0:04}_gt.png'.format(index), gt_np) 
            index += 1

            # plt.subplot(2, 2, 1)
            # plt.axis('off')
            # plt.title('RGB')
            # plt.imshow(input_rgb_np, cmap='plasma')
            # plt.subplot(2, 2, 2)
            # plt.axis('off')
            # plt.title('LowRes depth')
            # plt.imshow(input_depth_np, cmap='plasma')
            # plt.subplot(2, 2, 3)
            # plt.axis('off')
            # plt.title('Predicted depth')
            # plt.imshow(output_np, cmap='plasma')
            # plt.subplot(2, 2, 4)
            # plt.axis('off')
            # plt.title('GT')
            # plt.imshow(gt_np, cmap='plasma')
            # plt.show()
    
# Model evaluation on validation set
def evaluation(val_loader, model, criterion, device, args):
    print('Validation Start!')
    average_meter = AverageMetrics()
    criterion_l1 = criterion.l1_loss()

    model.eval()
    print(model.name)
    with tqdm(val_loader, unit='batch', bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as tepoch:
        for inputs in tepoch:
            input_rgb = inputs['rgb']
            input_depth = inputs['depth']
            target = inputs['gt']
            input_rgb, input_depth, target = input_rgb.to(device), input_depth.to(device), target.to(device)
            with torch.no_grad():
                if args.is_bin == 0:

                    torch.cuda.synchronize()
                    time_start = time.time()
                    outputs, features = model(input_rgb, input_depth)
                    torch.cuda.synchronize()
                    time_end = time.time()
                else:
                    torch.cuda.synchronize()
                    time_start = time.time()
                    bin_edges, outputs = model(input_rgb, input_depth)
                    torch.cuda.synchronize()
                    time_end = time.time()
                # loss = criterion_l1(outputs, target)

            datatime = time_end - time_start
            # print(1000.*datatime)
            metrics = Metrics()
            # metrics.evaluate(outputs, target, loss.item())
            metrics.evaluate(outputs, target, 0, datatime)
            # average_meter.update(loss.item(), datatime, metrics)
            # print(datatime)
            average_meter.update(metrics)
            average = average_meter.average()
            # print(average.time)

    print('\n**Evaluation Results**')
    #print('\tTotal loss: {:.2f}'.format(average.loss))
    print('\tAverage time: {:.2f} ms'.format(1000.*average.time))
    print('\tRMSE: {:.3f}'.format(average.rmse))
    print('\tREL: {:.3f}'.format(average.absrel))
    print('\tDelta1: {:.2f}'.format(100.*average.delta1))
    print('\tDelta2: {:.2f}'.format(100.*average.delta2))
    print('\tDelta3: {:.2f}'.format(100.*average.delta3))
    print('\tMax FPS: {:.2f}'.format(average.max_fps))
    print('\tAVG FPS: {:.2f}'.format(average.avg_fps))

if __name__=='__main__':
    torch.cuda.manual_seed(0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('Device: {}'.format(device))
    main()

