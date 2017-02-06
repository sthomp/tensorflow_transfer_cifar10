"""
Trying out the transfer learning example from:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
"""

from PIL import Image
import tensorflow as tf
import numpy as np
from sklearn import cross_validation
from data_utils import load_CIFAR10
from extract import create_graph, iterate_mini_batches, batch_pool3_features
from datetime import datetime
import matplotlib.pyplot as plt
from tsne import tsne
import seaborn as sns
import pandas as pd

#
# CIFAR10 Ops
# #

#flag to generate and save bottleneck features
DO_SERIALIZATION = False

def load_pool3_data():
    # Update these file names after you serialize pool_3 values
    X_test_file = 'X_test.npy'
    y_test_file = 'y_test.npy'
    X_train_file = 'X_train.npy'
    y_train_file = 'y_train.npy'
    return np.load(X_train_file), np.load(y_train_file), np.load(X_test_file), np.load(y_test_file)

def serialize_cifar_pool3(X,filename):
    print 'About to generate file: %s' % filename
    sess = tf.InteractiveSession()
    X_pool3 = batch_pool3_features(sess,X)
    np.save(filename,X_pool3)

def serialize_data():
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    serialize_cifar_pool3(X_train, 'X_train')
    serialize_cifar_pool3(X_test, 'X_test')
    np.save('y_train',y_train)
    np.save('y_test',y_test)


classes = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
cifar10_dir = 'resources/datasets/cifar-10-batches-py'

if DO_SERIALIZATION:
    serialize_data()
    exit(0)

X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_CIFAR10(cifar10_dir)
X_train_pool3, y_train_pool3, X_test_pool3, y_test_pool3 = load_pool3_data()
X_train, X_validation, Y_train, y_validation = cross_validation.train_test_split(X_train_pool3, y_train_pool3, test_size=0.20, random_state=42)


#
# Tensorflow stuff
# #

FLAGS = tf.app.flags.FLAGS
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'
BOTTLENECK_TENSOR_SIZE = 2048


tf.app.flags.DEFINE_integer('how_many_training_steps', 100,
                            """How many training steps to run before ending.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          """How large a learning rate to use when training.""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result',
                           """The name of the output classification layer in"""
                           """ the retrained graph.""")
tf.app.flags.DEFINE_integer('eval_step_interval', 10,
                            """How often to evaluate the training results.""")

graph = create_graph()
bottleneck_tensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME+':0')

#placeholder for bottleneck_features. Bottleneck layer itslef is not needed for training on new task. Only the stored bottleneck featuures are needed
X_Bottleneck = tf.placeholder(tf.float32,shape=[None, BOTTLENECK_TENSOR_SIZE])
#placeholder for true labels
Y_true = tf.placeholder(tf.float32,[None, 10],name= "Y_true")


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def ensure_name_has_port(tensor_name):
    """Makes sure that there's a port number at the end of the tensor name.
    Args:
      tensor_name: A string representing the name of a tensor in a graph.
    Returns:
      The input string with a :0 appended if no port was specified.
    """
    if ':' not in tensor_name:
        name_with_port = tensor_name + ':0'
    else:
        name_with_port = tensor_name
    return name_with_port

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def add_final_training_ops():
    """Adds a new softmax and fully-connected layer for training.
    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.
    The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
    Returns:
      Nothing.
    """

    #bottleneck_tensor = graph.get_tensor_by_name(ensure_name_has_port( BOTTLENECK_TENSOR_NAME))

    layer_weights = tf.Variable(
        tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, len(classes)], stddev=0.001),
        name='final_weights')
    layer_biases = tf.Variable(tf.zeros([len(classes)]), name='final_biases')

    logits = tf.add(tf.matmul(X_Bottleneck, layer_weights,name='final_matmul'),layer_biases,name="logits")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y_true)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean,logits

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def add_evaluation_step(Ylogits):
    """Inserts the operations we need to evaluate the accuracy of our results.
    Args:
      graph: Container for the existing model's Graph.
    Returns:
      Nothing.
    """
    correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_true, 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return evaluation_step

def encode_one_hot(nclasses,y):
    return np.eye(nclasses)[y]

def do_train(sess,X_input, Y_input, X_validation, Y_validation):
    mini_batch_size = 10
    n_train = X_input.shape[0]

    #graph = create_graph()

    train_step, cross_entropy, Ylogits = add_final_training_ops()

    init = tf.initialize_all_variables()
    sess.run(init)

    evaluation_step = add_evaluation_step(Ylogits)

    # Get some layers we'll need to access during training.

    i=0
    epocs = 1
    for epoch in range(epocs):
        shuffledRange = np.random.permutation(n_train)
        y_one_hot_train = encode_one_hot(len(classes), Y_input)
        y_one_hot_validation = encode_one_hot(len(classes), Y_validation)
        shuffledX = X_input[shuffledRange,:]
        shuffledY = y_one_hot_train[shuffledRange]

        for Xi, Yi in iterate_mini_batches(shuffledX, shuffledY, mini_batch_size):
            sess.run(train_step,
                     feed_dict={X_Bottleneck: Xi,
                                Y_true: Yi})
            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)

            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                  [evaluation_step, cross_entropy],
                  feed_dict={X_Bottleneck: Xi,
                             Y_true: Yi})
                validation_accuracy = sess.run(
                  evaluation_step,
                  feed_dict={X_Bottleneck: X_validation,
                             Y_true: y_one_hot_validation})
                print('%s: Step %d: Train accuracy = %.1f%%, Cross entropy = %f, Validation accuracy = %.1f%%' %
                    (datetime.now(), i, train_accuracy * 100, cross_entropy_value, validation_accuracy * 100))
            i+=1

    test_accuracy = sess.run(
        evaluation_step,
        feed_dict={X_Bottleneck: X_test_pool3,
                   Y_true: encode_one_hot(len(classes), y_test_pool3)})
    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

def show_test_images(sess,X_img, X_features, Y):
    n = X_img.shape[0]
    def rand_ordering():
        return np.random.permutation(n)

    def sequential_ordering():
        return range(n)

    for i in sequential_ordering():
        Xi_img=X_img[i,:]
        Xi_features=X_features[i,:].reshape(1,2048)
        result_tensor = sess.graph.get_tensor_by_name(ensure_name_has_port("logits"))
        probs = sess.run(result_tensor,
                         feed_dict={X_Bottleneck: Xi_features})
        predicted_class=classes[np.argmax(probs)]
        Yi = Y[i]
        Yi_label = classes[Yi]
        plt.title('true=%s, predicted=%s' % (Yi_label, predicted_class))
        ax = plt.axes()
        ax.grid(False)
        ax.imshow(Xi_img.astype('uint8'),interpolation='nearest')
        print Yi_label
        plt.show()

def plot_tsne(X_input_pool3, Y_input,n=10000):
    indicies = np.random.permutation(X_input_pool3.shape[0])[0:n]
    Y=tsne(X_input_pool3[indicies])
    num_labels=Y_input[indicies]
    labels=classes[num_labels]
    df=pd.DataFrame(np.column_stack((Y,num_labels,labels)), columns=["x1","x2","y","y_label"])
    sns.lmplot("x1","x2",data=df.convert_objects(convert_numeric=True),hue="y_label",fit_reg=False,legend=True,palette="Set1")
    print 'done'

# plot_tsne(X_train_pool3,y_train_pool3)
sess = tf.InteractiveSession()
do_train(sess,X_train,Y_train,X_validation,y_validation)
# show_test_images(sess,X_test_orig, X_test_pool3, y_test_orig)
