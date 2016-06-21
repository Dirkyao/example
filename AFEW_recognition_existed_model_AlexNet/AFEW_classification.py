import find_mxnet
import mxnet as mx
import os,sys
import argparse
# train_model haven't been finished yet
import train_model

def get_iterator(data_shape):
    def get_iterator_impl(args,kv):
        data_dir = args.data_dir
        flat = False if len(data_shape)==3 else True

        train           = mx.io.ImageRecordIter(
            path_imgrec = data_dir+'train.rec',
            data_shape  = data_shape,
            path_imglist= data_dir+'train.lst',
            batch_size  = 256
        )

        val             = mx.io.ImageRecordIter(
            path_imgrec = data_dir+'test.rec',
            data_shape  = data_shape,
            path_imglist= data_dir+'test.lst',
            batch_size  = 256
        )

        return(train,val)
    return get_iterator_impl


def get_AlexNet():
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3
    conv3 = mx.symbol.Convolution(
        data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    # stage 5
    fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
    # stage 6
    fc3 = mx.symbol.FullyConnected(data=dropout2, num_hidden=7)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax

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
    #third conv
    conv3 = mx.symbol.Convolution(data=pool2,kernel=(5,5),num_filter=128)
    pool3 = mx.symbol.Pooling(data=conv3,pool_type='max',kernel=(5,5),stride=(5,5))
    #first fullc
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten,num_hidden=1000)
    relu4 = mx.symbol.Activation(data=fc1,act_type='relu')
    #second fullc
    fc2 = mx.symbol.FullyConnected(data=relu4,num_hidden=50)
    #loss
    CNN = mx.symbol.SoftmaxOutput(data=fc2,name='softmax')
    return CNN

def parse_args():
    parser = argparse.ArgumentParser(description='Train an expression classifer on AFEW')
    parser.add_argument('--network',type = str,default='CNN',choices = ['CNN','RBM','RNN','LSTM'])
    parser.add_argument('--data-dir',type=str,default='Processed_data_v2/',help='the input data directory')
    parser.add_argument('--gpus',type = str,default='0,1',help='the gpus to be used, e.g "0,1,2,3 "')
    parser.add_argument('--num-examples',type=int,default=40518,help='the total number of training images')
    # This default argument needs to be updated
    parser.add_argument('--batch-size',type=int,default=256,help='the batch size')
    parser.add_argument('--model-prefix',type=str,help='the prefix of the model to load/save')
    parser.add_argument('--lr',type = float, default = 0.001,help='the initial learning rate')
    parser.add_argument('--save-model-prefix',type=str)
    parser.add_argument('--num-epochs',type=int,default=40,help='the number of training epochs')
    parser.add_argument('--lr-factor',type=float,default=1,help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    if args.network =='CNN':
        data_shape = (3,224,224)
        net = get_AlexNet()

    #train
        train_model.fit(args,net,get_iterator(data_shape))

