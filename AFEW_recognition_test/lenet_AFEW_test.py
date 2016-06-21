import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model

def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(4,4), stride=(4,4))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(8,8), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(4,4), stride=(4,4))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def get_iterator(data_shape):
    def get_iterator_impl(args,kv):
        data_dir = args.data_dir
        flat = False if len(data_shape)==3 else True

        train           = mx.io.ImageRecordIter(
            path_imgrec = data_dir+'train.rec',
            data_shape  = data_shape,
            path_imglist= data_dir+'train.lst',
            batch_size  = 128
        )

        val             = mx.io.ImageRecordIter(
            path_imgrec = data_dir+'test.rec',
            data_shape  = data_shape,
            path_imglist= data_dir+'test.lst',
            batch_size  = 128
        )

        return(train,val)
    return get_iterator_impl

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='lenet',
                        choices = ['mlp', 'lenet'],
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str, default='/home/dirkyao/Processed_data_test/',
                        help='the input data directory')
    parser.add_argument('--gpus', type=str, default='0',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=40517,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.0001,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.network == 'lenet':
        data_shape = (1,128,128)
        net = get_lenet()

    # train
    train_model.fit(args, net, get_iterator(data_shape))