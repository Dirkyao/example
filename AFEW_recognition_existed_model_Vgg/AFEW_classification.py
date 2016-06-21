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

def get_vgg(num_classes = 7):
    ## define alexnet
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mx.symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mx.symbol.Pooling(
        data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    pool5 = mx.symbol.Pooling(
        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # output
    fc8 = mx.symbol.FullyConnected(data=drop7, num_hidden=num_classes, name="fc8")
    softmax = mx.symbol.SoftmaxOutput(data=fc8, name='softmax')
    return softmax

def parse_args():
    parser = argparse.ArgumentParser(description='Train an expression classifer on AFEW')
    parser.add_argument('--network',type = str,default='VGG',choices = ['CNN','VGG','RNN','LSTM'])
    parser.add_argument('--data-dir',type=str,default='Processed_data_v2/',help='the input data directory')
    parser.add_argument('--gpus',type = str,default='0,1',help='the gpus to be used, e.g "0,1,2,3 "')
    parser.add_argument('--num-examples',type=int,default=40518,help='the total number of training images')
    # This default argument needs to be updated
    parser.add_argument('--batch-size',type=int,default=256,help='the batch size')
    parser.add_argument('--model-prefix',type=str,help='the prefix of the model to load/save')
    parser.add_argument('--lr',type = float, default = 0.001,help='the initial learning rate')
    parser.add_argument('--save-model-prefix',type=str)
    parser.add_argument('--num-epochs',type=int,default=20,help='the number of training epochs')
    parser.add_argument('--lr-factor',type=float,default=1,help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    if args.network =='VGG':
        data_shape = (3,224,224)
        net = get_vgg()

    #train
        train_model.fit(args,net,get_iterator(data_shape))

