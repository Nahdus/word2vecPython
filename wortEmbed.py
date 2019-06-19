# encoding=utf8
import tensorflow as tf
import numpy as np
import re

with open("./corpous.txt", encoding="utf8") as content:
    corpus_raw=content.read()

#corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
# corpus_raw ="hi how are you ? i am fine. thank you. "
# corpus_raw = corpus_raw.lower()

# convert to lower case

corpus_raw = corpus_raw.lower()

words = []
words=list(filter(lambda x : False if re.match("(\.|\?|\s+|\,|^$)",x) else True,re.split("(\.|\?|\s+|\,)",corpus_raw)))
# with open("wordsBeforeSet.txt","a") as file:
#     for each in words:
#         print(each+".")
#         file.write(each+"\n")






words = set(words) # so that all duplicate words are removed

# with open("wordsAfterSet.txt","a") as file:
#     for each in words:
#         file.write(each+"\n")

word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# print(int2word)
# print(word2int)
# raw sentences is a list of sentences.


raw_sentences=list(filter(lambda x : False if re.match("(\.|\?|^$)",x) else True, re.split("(\.|\?)",corpus_raw)))







sentences=[]
for sentence in raw_sentences:
    
    sentences.append(list(filter(lambda x: False if re.match("(\.|\?|\s+|\,|^$)",x) else True,re.split("(\.|\?|\s+|\,)",sentence))))




WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        # print(sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1])
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
            if nb_word != word:
                data.append([word, nb_word])



# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    print(vocab_size)
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    
    return temp

x_train = [] # input word
y_train = [] # output word
# print(data) 
# print(word2int)

for data_word in data:
    # print(data_word)
    print(vocab_size)
    # x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    # y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))
    
'''
# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# making placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5 # you can choose your own number
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!

# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iters = 10000
# train for n_iter iterations

for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

vectors = sess.run(W1 + b1)

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index


from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors) 

from sklearn import preprocessing

normalizer = preprocessing.Normalizer()
vectors =  normalizer.fit_transform(vectors, 'l2')

print(vectors)

import matplotlib.pyplot as plt


fig, ax = plt.subplots()
print(words)
for word in words:
    print(word, vectors[word2int[word]][1])
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
plt.show()

'''