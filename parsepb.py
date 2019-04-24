import tensorflow as tf
gf = tf.GraphDef()
gf.ParseFromString(open('object_detection_graph\\frozen_inference_graph.pb','rb').read())

print([n.name + '=>' +  n.op for n in gf.node if n.op in ( 'Softmax','Mul')])