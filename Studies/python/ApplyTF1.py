import tensorflow as tf
import numpy as np
# import ROOT
# import h5py
import time

import InputsProducer

model_file= 'prova_pipo_v2_par1.pb'

# file_name = 'GluGluToRadionToHHTo2B2Tau_M-750_narrow_tauTau_2017_ggHH_Res.root'

# data = InputsProducer.CreateRootDF(file_name, 0, True, True)
# X, Y, Z, var_pos, var_pos_z, var_name = InputsProducer.CreateXY(data, '../config/training_variables.json')

X = np.load('X_par1.npy')
# X= X[2929:2929+1, :, :]

# print(X)
# raise RuntimeError("stop")

with tf.gfile.GFile(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="HHModel")

    config = tf.ConfigProto(intra_op_parallelism_threads=2,
                            inter_op_parallelism_threads=2,
                            allow_soft_placement=True,
                            device_count = {'CPU' : 1, 'GPU' : 0})

    sess = tf.Session(graph=graph, config=config)

    #sess = tf.Session()


    #for n in graph.as_graph_def().node:
    #    print(n.name, n)
    # layer_names = [n.name for n in graph.as_graph_def().node]
    # print(layer_names)
    # for op in graph.get_operations():
    #     if op.name == 'HHModel/batch_normalization_rnn_0/batchnorm_1/mul_1':
    #         print(op)
    #         print("n_inputs={}".format(len(op.inputs)))
    #         for x in op.inputs:
    #             print(x)
    # raise RuntimeError("stop")
    x_graph = graph.get_tensor_by_name('HHModel/input:0')
    y_graph = graph.get_tensor_by_name('HHModel/NormToTwo/div:0')
    # y_graph = graph.get_tensor_by_name('HHModel/rnn_0/transpose_2:0')
    # y_graph = graph.get_tensor_by_name('HHModel/batch_normalization_post_12/batchnorm_1/add_1:0')
    # y_graph = graph.get_tensor_by_name('HHModel/output/Reshape_1:0')
    # y_graph = graph.get_tensor_by_name('HHModel/slice/strided_slice:0')

    #print(y_graph.shape)

    sess.run(tf.global_variables_initializer())


    # pred = sess.run(y_graph, feed_dict={ x_graph: np.ones((4, 10, 15))})
    start = time.time()
    pred = sess.run(y_graph, feed_dict={ x_graph: X})
    end = time.time()
    print('Passed time:',end - start)

    #print('type pred=', pred.shape)

    #pred = pred.reshape(pred.shape[0:2])
    # print(np.array2string(pred, separator=','))
    #print(np.sum(pred, axis=-1))

    np.save('pred_TF1_par1', pred)


    # h5f = h5py.File('pred.h5', 'w')
    # h5f.create_dataset('pred', data=pred)
    # h5f.close()
