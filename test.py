# ==================================================
# Load and test a trained model on Traing Set.
# ==================================================
import tensorflow as tf
import numpy as np
import preprocess
from model import GRUModel
from tensorflow.contrib import learn

from config import *

print("Restoring Vocab...")
test_data = preprocess.test_data(dataset)
c = [y for x in test_data[2] for y in x]
max_len = max([len(x) for x in c], 374)
vocab = learn.preprocessing.VocabularyProcessor(max_len)
vocab.fit(c)

qq, ll, cc = zip(*[[test_data[0][i], test_data[1][i], test_data[2][i]] for i in range(len(test_data[0]))])

print("Converting text to data...")
train_q = preprocess.build_vocab(qq, vocab)
train_c = [preprocess.build_vocab(x, vocab) for x in cc]
train_l = [preprocess.build_vocab(x, vocab) for x in ll]

def generate_dataset():
    result = []
    for i, c in enumerate(train_c):
        q = train_q[i]
        chioces = train_l[i]
        result.append([c, q, train_l[i]])
    return result

test_data = generate_dataset()
ic_sents, iq, chioces = zip(*test_data)

print(" [*] Max sentence length: {:d}".format(max_len))
print(" [*] Vocabulary Size: {:d}".format(len(vocab.vocabulary_)))
print(" [*] Test Question Size: {:d}".format(len(test_data)))

# ==================================================
print("Restoring Model...")
input_c = tf.placeholder(tf.float32, [None, 1, max_len], name="ic")
input_q = tf.placeholder(tf.float32, [None, 1, max_len], name="iq")
input_r = tf.placeholder(tf.float32, [None, 1, max_len], name="ir")
input_w = tf.placeholder(tf.float32, [None, 1, max_len], name="iw")
state = tf.placeholder(tf.float32, [None, max_len], name="state")
dropout = tf.placeholder(tf.float32, name="dropout")

zero_input = [np.random.randn(1, max_len) for _ in range(batch_size)]
zero_state = [np.random.randn(max_len) for _ in range(batch_size)]
costs = []

sess = tf.Session()
model = GRUModel(input_c, input_q, input_r, input_w, state, dropout, num_hidden=max_len)
model.load(sess, save_dir='save', dataset=dataset)

# ==================================================
def encode(c_batch, q_batch):
    def merge(article, question):
        prev = zero_state
        for sent in article:
            prev = sess.run(model.prediction, {
                input_c: [sent],
                input_q: [question],
                input_r: zero_input, input_w: zero_input, state: prev, dropout: 0})
            return prev
    assert len(c_batch) == len(q_batch), "Must input same bacth size of context and question"
    encode_batch = [merge(c, q) for c, q in zip(c_batch, q_batch)]
    return encode_batch

# ==================================================
num_correct = 0

f = open('answer.txt', 'w+')

print("Running Model...")
for epoch in range(len(test_data)):
    idx = epoch
    # generate batches
    batch_iq = [[iq[idx]]] * batch_size
    # batch_answers
    answers = chioces[idx]
    batch_ans1 = [[answers[0]]] * batch_size
    batch_ans2 = [[answers[1]]] * batch_size
    batch_ans3 = [[answers[2]]] * batch_size
    batch_ans4 = [[answers[3]]] * batch_size
    # batch_context
    c_batch = [ic_sents[idx]] * batch_size
    c_batch = [[[y] for y in x] for x in c_batch]
    # encode context & question for all context sentences
    batch_enc = encode(c_batch, batch_iq)

    sims = []
    for x in (batch_ans1, batch_ans2, batch_ans3, batch_ans4):
        sims.append(sess.run(model.cosine_cost, {
            input_c: zero_input, input_q: zero_input, input_r: x, input_w: x, state: batch_enc[0], dropout: 1}))
    best_ans = sims.index(min(sims))
    f.write("%d\n" % best_ans)

f.close()
