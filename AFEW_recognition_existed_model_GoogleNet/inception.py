import find_mxnet
import mxnet as mx
import logging
import os
import scipy.io

def fit(args,data_loader,batch_end_callback=None):
    #kvstore
    kv=mx.kvstore.create(args.kv_store)

    #logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'

    logging.basicConfig(level=logging.DEBUG,format=head)
    logging.info('start with arguments %s',args)

    #load model
    model_prefix = "Inception/Inception_BN"

    #model_args = {}

    num_round = 39
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    model = mx.model.FeedForward.load(model_prefix,num_round,ctx=devs,num_epoch = args.num_epochs+num_round,learning_rate = args.lr,momentum = 0.9,wd = 0.00001,numpy_batch_size = 256,epoch_size = args.num_examples / args.batch_size)

    # modify the structure of the google net
    internals = model.symbol.get_internals()
    preserve_symbol = internals["flatten_output"]
    fc_last = mx.symbol.FullyConnected(data=preserve_symbol,num_hidden=7)
    softmax = mx.symbol.SoftmaxOutput(data=fc_last,name = 'softmax')
    AFEW_symbol = softmax
    AFEW_model = mx.model.FeedForward(
        ctx = devs,
        symbol = AFEW_symbol,
        num_epoch = args.num_epochs+num_round,
        learning_rate = args.lr,
        momentum = 0.9,
        wd = 0.00001,
        numpy_batch_size = 256,
        arg_params=model.arg_params, 
        aux_params=model.aux_params,
        allow_extra_params=True)

    # visualize the structure of the network
    #my_symbol = mx.viz.plot_network(model.symbol, shape={"data": (1, 3, 224, 224)})
    #my_symbol.view();

    #data
    (train,val) = data_loader(args,kv)

    #train
    epoch_size = args.num_examples / args.batch_size


    # model = mx.model.FeedForward(
    #     ctx         =devs,
    #     symbol      =network,
    #     num_epoch   =args.num_epochs,
    #     learning_rate=args.lr,
    #     momentum    =0.9,
    #     wd          =0.00001,
    #     initializer = mx.init.Xavier(factor_type='in',magnitude=2.34)
    # )

    #just get the accuracy rather than top-k-accuracy.
    eval_metrics = ['accuracy']
    #accuracy
    eval_metrics.append(mx.metric.create('Accuracy'))

    batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))

    AFEW_model.fit(
        X           = train,
        eval_data   = val,
        eval_metric = eval_metrics,
        kvstore     = None,
        batch_end_callback  = batch_end_callback,
        epoch_end_callback  = None
    )

    #Y = model.predict(
    #    X=val,
    #    return_data=True
    #)

    #Z = model.predict(
    #    X=train,
    #    return_data=True
    #)

    #scipy.io.savemat('train_feature.mat', {'train_feature': Z[0]});
    #scipy.io.savemat('test_feature.mat', {'test_feature': Y[0]});

    #scipy.io.savemat('train_label.mat', {'train_label': Z[2]});
    #scipy.io.savemat('test_label.mat', {'test_label': Y[2]});
    print 'done'


