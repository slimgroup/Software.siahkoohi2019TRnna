import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import LearnedWaveSim

parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment_dir', dest='experiment_dir', default='dispersion', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=200, help='scale images to this size')
parser.add_argument('--image_size0', dest='image_size0', type=int, default=481, help='then crop to this size')
parser.add_argument('--image_size1', dest='image_size1', type=int, default=381, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./log', help='sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--correction_num', dest='correction_num', type=int, default=3, help='number of networks')
parser.add_argument('--same_model_training', dest='same_model_training', type=int, default=0, help='do same_model_training learning')
parser.add_argument('--training_fraction', dest='training_fraction', type=int, default=2, help='the fraction of training data to use')
parser.add_argument('--netEpoch', dest='netEpoch', type=int, default=5, help='epochs per network')
parser.add_argument('--data_path', dest='data_path', type=str, default='/home/ec2-user/data/', help='path to velocity model')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = LearnedWaveSim(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)

if __name__ == '__main__':
    tf.app.run()
