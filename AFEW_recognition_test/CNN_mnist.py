import find_mxnet
import mxnet as mx
import os,sys
import argparse
# train_model haven't been finished yet
import train_model_2

def get_iterator(data_shape):
    def get_iterator_impl(args, kv):
        data_dir = args.data_dir

        flat = False if len(data_shape) == 3 else True

        train = mx.io.MNISTIter(
            image=data_dir + "train-images-idx3-ubyte",
            label=data_dir + "train-labels-idx1-ubyte",
            input_shape=data_shape,
            batch_size=args.batch_size,
            shuffle=True,
            flat=flat,
            num_parts=kv.num_workers,
            part_index=kv.rank)

        val = mx.io.MNISTIter(
            image=data_dir + "t10k-images-idx3-ubyte",
            label=data_dir + "t10k-labels-idx1-ubyte",
            input_shape=data_shape,
            batch_size=args.batch_size,
            flat=flat,
            num_parts=kv.num_workers,
            part_index=kv.rank)

        return (train, val)
    return get_iterator_impl

def get_CNN():
    data = mx.symbol.Variable('data')
    #first conv
    conv1 = mx.symbol.Convolution(data = data,kernel=(5,5),num_filter=64)
    relu1 = mx.symbol.Activation(data=conv1,act_type='relu')
    pool1 = mx.symbol.Pooling(data=relu1,pool_type='max',kernel=(2,2),stride=(2,2))
    #second conv
    conv2 = mx.symbol.Convolution(data=pool1,kernel=(5,5),num_filter=64)
    relu2 = mx.symbol.Activation(data=conv2,act_type='relu')
    pool2 = mx.symbol.Pooling(data=relu2,pool_type='max',kernel=(2,2),stride=(2,2))
    #first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten,num_hidden=500)
    relu4 = mx.symbol.Activation(data=fc1,act_type='relu')
    #second fullc
    fc2 = mx.symbol.FullyConnected(data=relu4,num_hidden=50)
    #loss
    CNN = mx.symbol.SoftmaxOutput(data=fc2,name='softmax')
    return CNN

def parse_args():
    parser = argparse.ArgumentParser(description='Train an expression classifer on AFEW')
    parser.add_argument('--network',type = str,default='CNN',choices = ['CNN','RBM','RNN','LSTM'])
    parser.add_argument('--data-dir',type=str,default='mnist/',help='the input data directory')
    parser.add_argument('--gpus',type = str,default='0',help='the gpus to be used, e.g "0,1,2,3 "')
    parser.add_argument('--num-examples',type=int,default=60000,help='the total number of training images')
    # This default argument needs to be updated
    parser.add_argument('--batch-size',type=int,default=512,help='the batch size')
    parser.add_argument('--model-prefix',type=str,help='the prefix of the model to load/save')
    parser.add_argument('--lr',type = float, default = 0.001,help='the initial learning rate')
    parser.add_argument('--save-model-prefix',type=str)
    parser.add_argument('--num-epochs',type=int,default=10,help='the number of training epochs')
    parser.add_argument('--lr-factor',type=float,default=1,help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--lr-factor-epoch', type=float, default=1, help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    if args.network == 'CNN':
        data_shape = (1,128,128)
        net = get_CNN()

    #train
    train_model_2.fit(args, net, get_iterator(data_shape))