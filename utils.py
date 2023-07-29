import argparse

def parse_commands(mode):
    parser = argparse.ArgumentParser()
    if mode=='train':
        
        parser.add_argument('-isbin', '--is_bin', type=int, default=0, help='Is_bin')
        parser.add_argument('-lowRes', '--lowRes', type=int, default=1, help='Low resolution input')
        parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('-ep', '--epochs', type=int, default=50, help='Number of Epochs')
        parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('-wk', '--workers', type=int, default=24, help='Workers')
        parser.add_argument('-bin', '--bins', type=int, default=4, help='Number of bins')
        parser.add_argument('-fz', '--freeze', type=bool, default=False, help='Freeze')
        parser.add_argument('-ckp', '--check_point', default='./result/test/', help='Path of the check point weights')
        parser.add_argument('-data', '--data', type=str, default='nyudepthv2', help='Datasets')

        # Weights
        parser.add_argument('-pw', '--pixel_weight', type=float, default=1, help='Weight of pixel loss')
        parser.add_argument('-gw', '--grad_weight', type=float, default=1, help='Weight of grad loss')
        parser.add_argument('-bw', '--bin_weight', type=float, default=1, help='Weight of bin loss')
        parser.add_argument('-sw', '--ssim_weight', type=float, default=1, help='Weight of ssim loss')

        args = parser.parse_args()

    elif mode=='eval':
        parser.add_argument('-isbin', '--is_bin', type=int, default=0, help='Is_bin')
        parser.add_argument('-lowRes', '--lowRes', type=int, default=1, help='Low resolution input')
        parser.add_argument('-m', '--mode', default='e', help='Mode')
        parser.add_argument('-w', '--weight', default='./result/test/ep50_97.65_0.218.pth', help='Path of the weights')
        parser.add_argument('-n', '--number', default=0, type=int, help='Number of images')
        parser.add_argument('-d', '--device', default='cpu', help='Device')
        parser.add_argument('-wk', '--workers', type=int, default=24, help='Workers')
        parser.add_argument('-data', '--data', type=str, default='nyudepthv2', help='Datasets')

        args = parser.parse_args()

    return args
