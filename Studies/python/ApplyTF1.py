import tensorflow as tf
import numpy as np
import ROOT

model_file= 'prova_pipo.pb'

file_name = 'GluGluToRadionToHHTo2B2Tau_M-750_narrow_tauTau_2017_ggHH_Res.root'

file = ROOT.TFile.Open(file_name, 'READ')

with tf.gfile.GFile(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="HHModel")
    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())

    #for n in graph.as_graph_def().node:
    #    print(n.name, n)
    # layer_names = [n.name for n in graph.as_graph_def().node]
    # print(layer_names)
    #raise RuntimeError("stop")
    x_graph = graph.get_tensor_by_name('HHModel/input:0')
    y_graph = graph.get_tensor_by_name('HHModel/NormToTwo/div:0')

    print(y_graph.shape)


    pred = sess.run(y_graph, feed_dict={ x_graph: np.ones((4, 10, 15))})
    #pred = pred.reshape(pred.shape[0:2])
    print(pred)
    print(np.sum(pred, axis=-1))
