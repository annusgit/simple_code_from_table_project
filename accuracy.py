
import argparse
import numpy as np 
import tensorflow
print('log: imports successful!')


def main():
    parser = argparse.ArgumentParser(description='test accuracy on table detection problem')
    parser.add_argument('--cf', '--checkpoint_folder', type=str, 
        dest='checkpoint_directory', help='checkpoints directory path')

    args = parser.parse_args()
    ckpt_dir = args.checkpoint_directory
    print('path: {}'.format(ckpt_dir))    

    saver = tf.train.Saver()

if __name__ == '__main__':
    print('log: entering main routine')
    main()












