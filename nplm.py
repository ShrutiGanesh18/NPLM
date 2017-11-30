
import tensorflow as tf
import numpy as np
import pickle
import time
import argparse

"""
    Returns the onehot vectors for context words and the target word.
"""
def create_onehot(words,dictionary,context_size,vocab_size):
    onehot_context = []
    global index
    for word in words[index:index+context_size]:
        onehot_context.append(np.eye(vocab_size)[dictionary[word]].reshape((1,vocab_size)))
    onehot_target = np.eye(vocab_size)[dictionary[words[index+context_size]]].reshape((1,vocab_size))
    index = index + 1
    return onehot_context, onehot_target
"""
    Returns the words and dictionary that were created by the preprocessing module.
"""
def create_dictionary():
    with open("word_dump",'rb') as f:
        words = pickle.load(f)
    with open("dict_dump",'rb') as f:
        dictionary = pickle.load(f)
    return words, dictionary
"""
    Implements NPLM
"""
def start_model():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Train a Neural Probabilistic Language Model')
    parser.add_argument('-embedding_dim',nargs='?',dest='embedding_dim',default=50,help='Give the size of the embedding. Default 50')
    parser.add_argument('-hidden_dim',nargs='?',dest='hidden_dim',default=100,help='Give the size of the hidden layer of the model. Default 100')
    parser.add_argument('-context_size',nargs='?',dest='context_size',default=5,help='Give the size of the context window. Default 5')
    parser.add_argument('-learning_rate',nargs='?',dest='learning_rate',default=0.01,help='Give the learning rate of the model. Default 0.01')

    args_parse = parser.parse_args()
    embedding_dim = args_parse.embedding_dim
    hidden_dim = args_parse.hidden_dim
    context_size = args_parse.context_size
    learning_rate = args_parse.learning_rate

    # setting up dictionary and input text
    words, dictionary = create_dictionary()
    vocab_size = len(dictionary)
    input_dim = vocab_size
    output_dim = vocab_size

    # setting input and output variables
    input_X = []
    for i in range(context_size):
        input_X.append(tf.placeholder(tf.float32,shape=[None,input_dim],name="X_"+str(i)))
    actual_Y = tf.placeholder(tf.float32,shape=[None,output_dim],name="Y")

    # setting weight parameters
    C = tf.get_variable('C',shape=[vocab_size,embedding_dim],initializer=tf.random_normal_initializer())
    W1 = tf.get_variable('W_1',shape=[context_size*embedding_dim,hidden_dim],initializer=tf.random_normal_initializer())
    b1 = tf.Variable(tf.constant(0.1,shape=[hidden_dim]),name='b1')
    W2 = tf.get_variable('W_2',shape=[hidden_dim,output_dim],initializer=tf.random_normal_initializer())
    b2 = tf.Variable(tf.constant(0.1,shape=[output_dim]),name='b2')

    # setting up the graph operations
    I = []
    for x in input_X:
        I.append(tf.matmul(x,C))
    I_concat = tf.concat(1,I)

    # setting up hidden layer
    H1 = tf.matmul(I_concat,W1)+b1
    Z1 = tf.tanh(H1)

    # setting up output layer
    H2 = tf.matmul(Z1,W2)+b2

    # setting up loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(H2,actual_Y))
    updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # initiating saver
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        with open("C_init","wb") as f:
            np.savetxt(f, C.eval())

        global index
        index = 0

        index_last = len(words)-context_size
        while (index < index_last):
            onehot_context,onehot_target = create_onehot(words,dictionary,context_size,vocab_size)
            print index
            dict_feed = dict()
            for i in range(context_size):
                dict_feed[input_X[i]] = onehot_context[i]
            dict_feed[actual_Y] = onehot_target
            session.run([updates],feed_dict=dict_feed)

            if index%10000 == 0:
                with open("C_done"+str(index),"wb") as f:
                    np.savetxt(f, C.eval())
                save_path = saver.save(session, "model.ckpt")

        end_time = time.time()
        print str(start_time) + " to " + str(end_time)


if __name__=="__main__":
    start_model()
