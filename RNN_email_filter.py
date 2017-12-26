import email
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import NLP_module
import html2text
from nltk.corpus import stopwords
ops.reset_default_graph()

"""
https://github.com/nfmcclure/tensorflow_cookbook/blob/master/09_Recurrent_Neural_Networks/02_Implementing_RNN_for_Spam_Prediction/02_implementing_rnn.py
"""

# Create a text cleaning function
def clean_text(text_string): 
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.escape_snob = True
    txt = h.handle(text_string)
    txt = txt.lower()
    p = re.compile('\W+')
    splits  = p.split(txt)
    print(splits)
    result = ""
    start_to_parse = False
    for word in splits:
        if NLP_module.strCmp(word, "Date"):
            start_to_parse = True
        if start_to_parse == False:
            continue
        found = re.search('[0-9]+', word)
        if found != None:
            continue
        stop_wds = stopwords.words("english")
        stop_wds.extend(["email", "www", "com", "http", "html", "gif", "smtp", "sender", "received", "zzzz", "yyyy","localhost", "org", "esmtp", "debian", "return", "path"])
        if word in stop_wds:
            continue
        if len(word) <= 2 or len(word) >= 10:
            continue
        result = result + " " + word
    text_string = result.lower()
    return(text_string)

# Start a graph
sess = tf.Session()

# Set RNN parameters
epochs = 50
batch_size = 200
max_sequence_length = 100 ## 100
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)

labels = dict()
label_path = '/Users/lixiaodan/Desktop/ece590/CSDMC2010_SPAM/CSDMC2010_SPAM/SPAMTrain.label'
infile = open(label_path,'r')
label_List = list()
for line in infile:
    tp = line.split(" ")[1]
    eml_name = tp.split("\n")[0]
    labels[eml_name] = line.split(" ")[0]
    label_List.append(line.split(" ")[0])
infile.close()

path = '/Users/lixiaodan/Desktop/ece590/CSDMC2010_SPAM/CSDMC2010_SPAM/training_new'
listing = os.listdir(path)
listing = listing

fail_IO = list()
gd_cnt = 0
bad_cnt = 0
text_target = list()
text_data_train = list()

for i in range(len(listing)):
    fle = listing[i]
    if str.lower(fle[-3:])=="eml":
        try:
            msg = email.message_from_file(open(path + '/' + fle))
            strs = msg.as_string()
            cleantext = clean_text(strs)
            #print(cleantext)
            text_data_train.append(cleantext)
            if labels[fle] == "1":
                gd_cnt = gd_cnt + 1
                text_target.append(1)
            else:
                bad_cnt = bad_cnt + 1
                text_target.append(0)
        except UnicodeDecodeError:
            fail_IO.append(fle)
            continue
# Clean texts
#text_data_train = [clean_text(x) for x in text_data_train]

# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                     min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))

# Shuffle and split data
text_processed = np.array(text_processed)
text_data_target = np.array(text_target)
#text_data_target = np.array([1 if x=='ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled)*0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# Create placeholders
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

# Create embedding
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
#embedding_output_expanded = tf.expand_dims(embedding_output, -1)

# Define the RNN cell
#tensorflow change >= 1.0, rnn is put into tensorflow.contrib directory. Prior version not test.
if tf.__version__[0]>='1':
    cell=tf.contrib.rnn.BasicRNNCell(num_units = rnn_size)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)

output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)


weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.matmul(last, weight) + bias

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output) # logits=float32, labels=int32
loss = tf.reduce_mean(losses)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

# prediction result
pred_correction = tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64))
prediction = tf.argmax(logits_out, 1)

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
test_pred = list()
test_predcor = list()
# Start training
for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train)/batch_size) + 1
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        
        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)
        
    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    
    # Run Eval Step
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc, temp_test_predcor, temp_test_prediction = sess.run([loss, accuracy, pred_correction, prediction], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    test_predcor.append(temp_test_predcor)
    test_pred.append(temp_test_prediction)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
    
# Plot loss over time
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

# get the RoC curve
predcor = test_predcor[-1]
#print(predcor)
prediction = test_pred[-1]
#print(prediction)
NLP_module.plotRoc(prediction, y_test)

##1 stands for a HAM and 0 stands for a SPAM
detec_spam = 0
non_detec_spam = 0
detec_good = 0
non_detec_good = 0
for i in range(len(prediction)):
    if predcor[i] == True:
        if y_test[i] == 0:
            detec_spam = detec_spam + 1
        else:
            detec_good = detec_good + 1
    if predcor[i] == False:
        if y_test[i] == 1:
            non_detec_good = non_detec_good + 1
        else:
            non_detec_spam = non_detec_spam + 1
total_spam = detec_spam + non_detec_spam
total_good = detec_good + non_detec_good
spam_rate = 1.0 * detec_spam / (detec_spam + non_detec_spam)
good_rate = 1.0 * detec_good / (detec_good + non_detec_good)
accuracy = 1.0 * (detec_spam + detec_good) / (len(prediction))
print("Total spam email")
print(total_spam)
print("Total good email")
print(total_good)
print("Spam rate is")
print(spam_rate)
print("Good rate is")
print(good_rate)
print("Accuracy")
print(accuracy)