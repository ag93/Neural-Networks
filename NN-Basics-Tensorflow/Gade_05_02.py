# Gade, Aniket
# 1001-505-046
# 2018-11-26
# Assignment-05-02
import sklearn.datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

display_step = 1
random_weights = False
def generate_data(dataset_name, n_samples, n_classes):
    if dataset_name == 'swiss_roll':
        data = sklearn.datasets.make_swiss_roll(n_samples, noise=1.5, random_state=99)[0]
        data = data[:, [0, 2]]
    if dataset_name == 'moons':
        data = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.15)[0]
    if dataset_name == 'blobs':
        data = sklearn.datasets.make_blobs(n_samples=n_samples, centers=n_classes*2, n_features=2, cluster_std=0.85*np.sqrt(n_classes), random_state=100)
        X,y = data[0]/10., [i % n_classes for i in data[1]]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return(X, onehot_encoded)
    if dataset_name == 's_curve':
        data = sklearn.datasets.make_s_curve(n_samples=n_samples, noise=0.15, random_state=100)[0]
        data = data[:, [0,2]]/3.0

    ward = AgglomerativeClustering(n_clusters=n_classes*2, linkage='ward').fit(data)
    X,y =  data[:]+np.random.randn(*data.shape)*0.03, [i % n_classes for i in ward.labels_]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return(X, onehot_encoded)
    
def nn_model(data, n_node_hl, n_classes, activation_type):
    hidden_layer = {'weights': tf.Variable(tf.random_normal([2, n_node_hl], seed = 1, stddev = 0.001)), 'biases': tf.Variable(tf.random_normal([n_node_hl], seed = 12, stddev = 0.001))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_node_hl, n_classes], seed = 2, stddev = 0.001)), 'biases': tf.Variable(tf.random_normal([n_classes], seed = 21, stddev = 0.001))}

    l1 = tf.add(tf.matmul(data, hidden_layer['weights']), hidden_layer['biases'])
    
    if(activation_type == 'Sigmoid'):
        l1 = tf.nn.sigmoid(l1)
    else:
        l1 = tf.nn.relu(l1)
    
    output = tf.matmul(l1, output_layer['weights'])    
    return(output, output_layer['weights'])
    
def train_nn(points, labels, lr, n_node_hl, n_classes, activation_type, alpha, random_weights):
    x = tf.placeholder('float',[None, 2])
    y = tf.placeholder('float')
#    display_step = 10
    prediction, weights = nn_model(points, n_node_hl,n_classes, activation_type)
    
    regularizer = tf.nn.l2_loss(weights)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    loss = tf.reduce_mean(cost + alpha * regularizer)
    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(loss)
    
    if (random_weights):
        num_epochs = 1
    else:
        num_epochs = 10
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            _, epoch_loss = sess.run([optimizer, cost], feed_dict = {x:points, y:labels})
            if(epoch % display_step == 0):
                print("Epoch", epoch+1, "of", num_epochs, "Loss = ", epoch_loss)
        array = tf.arg_max(prediction, 1)
        pred = array.eval()
        correct = tf.equal( tf.arg_max(prediction, 1), tf.arg_max(y, 1))   
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy = ", accuracy.eval({x:points, y:labels}))
    return pred

def main(points, labels, classes, samples, hidden_nodes, activation_type, learning_rate, alpha, random_weights):
#    print(classes, samples, hidden_nodes, activation_type, learning_rate, alpha)
    n_node_hl = hidden_nodes
    n_classes = classes
    points = np.array(points, dtype="float32")
    pred = train_nn(points, labels, learning_rate, n_node_hl, n_classes, activation_type, alpha, random_weights)
    return pred
    
if __name__ == "__main__" :
    n_node_hl = 100
    n_classes = 4
    n_samples = 200
    learning_rate = 0.1
    alpha = 0.1
    points, labels = generate_data('blobs', n_samples, n_classes)
    points = np.array(points, dtype="float32")
    pred = train_nn(points, labels, learning_rate, n_node_hl, n_classes, activation_type = "Relu", alpha = 0.1, random_weights = False)
