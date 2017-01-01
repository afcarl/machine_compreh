# ==================================================
# Load and test a trained model on Traing Set.
# ==================================================
import tensorflow as tf
import numpy as np
import preprocess
from model import GRUModel
from tensorflow.contrib import learn

from tqdm import tqdm

from config import *

print("Loading text data...")
qq, ll, cc, aa = preprocess.train_data(dataset)

words = preprocess.unique_words([cc, qq])
print('= Unique words: {:d}'.format(len(words)))

vec = preprocess.read_glove(words) # {key: word_vec}
print('= Found vectors: {:d}'.format(len(vec)))

vec_len = len(vec['the'])
print('= Vec len: {:d}'.format(vec_len))

print("Text to vec...")
vq = [preprocess.text2vec(x, vec) for x in qq] # [[vec, ..], [vec, ...], ...], each [vec, ...] is a sentence
vc = [preprocess.text2vec(x, vec) for x in cc] # [[vec, ..], [vec, ...], ...], each [vec, ...] is a paragraph
vl = [[preprocess.text2vec(y, vec) for y in x] for x in ll] # [[vec, ..], [vec, ...], [vec, ...], [vec, ...]]
va = [vl[i][x-1] for i, x in enumerate(aa)]

max_len = max([len(x) for x in vc])
set_len = len(vc)
print('= Max len: {:d}'.format(max_len))

del qq, ll, cc, words

print('= Max sentence length: {:d}'.format(max_len))
print('= Train Question Size: {:d}'.format(set_len))

# ==================================================
print("Rstoring Model...")
va = [y for x in vl for y in x]
row_size, ans_len = vec_len, max([len(x) for x in va])
input_c = tf.placeholder(tf.float32, [None, 1, row_size], name="ic") # word of vc
input_q = tf.placeholder(tf.float32, [None, 1, row_size], name="iq") # word of vq
input_r = tf.placeholder(tf.float32, [None, ans_len, row_size], name="ir")
input_w = tf.placeholder(tf.float32, [None, ans_len, row_size], name="iw")
state = tf.placeholder(tf.float32, [None, row_size], name="state")
dropout = tf.placeholder(tf.float32, name="dropout")

zero_parah = np.random.randn(1, row_size)
zero_input = np.random.randn(ans_len, row_size)
zero_state = np.random.randn(row_size)

def create_batch(tensor, batch_size):
    return [tensor] * batch_size

batch_zero_parah = create_batch(zero_parah, batch_size)
batch_zero_input = create_batch(zero_input, batch_size)
batch_zero_state = create_batch(zero_state, batch_size)

costs = []

sess = tf.Session()
model = GRUModel(input_c, input_q, input_r, input_w, state, dropout, num_hidden=vec_len)
model.load(sess, save_dir='save', dataset=dataset)

# ==================================================
def encode(v, q):
    prev = batch_zero_state
    for x in q:
        batch_q = create_batch([x], batch_size) # each word from vq
        for y in v:
            batch_w = create_batch([y], batch_size) # each word from vc
            prev = sess.run(model.prediction, {
                input_c: batch_w,
                input_q: batch_q,
                input_r: batch_zero_input,
                input_w: batch_zero_input,
                state: batch_zero_state,
                dropout: 0
            })
    return prev

def pad_zero(vv, max_len):
    if max_len < len(vv):
        return vv
    vv += [[0] * vec_len] * (max_len - len(vv))
    return vv

vl = [[pad_zero(y, ans_len) for y in x] for x in vl]


# ==================================================
print('Running Model...')
print('= Drop Probability: %1.2f' % drop_prob)
print('= Batch Size: %d' % batch_size)
print('= Max Epoch: %d' % max_epoch)

n_correct1 = 0
n_correct2 = 0

for i in tqdm(range(300)):
    sims = []
    truth = int(aa[i])
    batch_cq = encode(vc[i], vq[i])

    for opt in vl[i]:
        batch_ir = create_batch(opt, batch_size)
        sims.append(sess.run(model.evaluate, {
            input_c: batch_zero_parah,
            input_q: batch_zero_parah,
            input_r: batch_ir,
            input_w: batch_ir,
            state: batch_cq,
            dropout: 1
        }))
    sims = [x-np.floor(sims[0]) for x in sims]
    guess1 = sims.index(min(sims))
    guess2 = sims.index(max(sims))
    if guess1 == truth: n_correct1 += 1
    if guess2 == truth: n_correct2 += 1
    print('[{:3d}] guess: {:d}, guess2: {:d}, truth: {:d}, {:3d}, {:3d}, sims:'.format(i+1, guess1, guess2, truth, n_correct1, n_correct2), sims)

print('Guess1: {:2.2f}%'.format(n_correct1 * 100 / 300))
print('Guess2: {:2.2f}%'.format(n_correct2 * 100 / 300))
