import argparse
# Training settings
parser = argparse.ArgumentParser(description='dorefa-net implementation')

#########################
#### data parameters ####
#########################
parser.add_argument("--data_name", type=str, default="wiki", # wiki xmedianet2views nus inria
                    help="data name")
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--log_name', type=str, default='noisylabel')
parser.add_argument('--ckpt_dir', type=str, default='noisylabel')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--eval_batch_size', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--ls', type=str, default='cos', help='lr scheduler')
parser.add_argument('--loss', type=str, default='CE', help='CE RCE MAE') # MCE
parser.add_argument('--output_dim', type=int, default=512, help='output shape')
parser.add_argument('--noisy_ratio', type=float, default=0.6) # 0.2 0.4 0.6 0.8
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--tau', type=float, default=1.)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--views', nargs='+', help='<Required> Quantization bits', default=['Img', 'Txt', 'Audio', '3D', 'Video']) #Img, Txt, Audio, 3D, Video

args = parser.parse_args()
print(args)

# --max_epochs 100 --log_name noisylabel_se --loss CE  --lr 0.05 --train_batch_size 50 --beta 1
# --max_epochs 50 --log_name noisylabel_mce --loss MCE  --lr 0.05 --train_batch_size 50 --beta 0.7 --noisy_ratio 0.2 --data_name wiki
# --max_epochs 50 --log_name noisylabel_mce --loss MCE  --lr 0.05 --train_batch_size 50 --beta 0.4 --noisy_ratio 0.6 --data_name wiki