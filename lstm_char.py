import os
import re
import urllib.request
import numpy as np
import time
import tensorflow as tf

metamorphosis = "metamorphosis.txt"
url = "http://www.gutenberg.org/cache/epub/5200/pg5200.txt"

batch_size = 60         # Sequences per batch
num_steps = 30          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability
epochs = 40
save_every_n = 100      # Save every N iterations

""" params
batch_size - Number of sequences running through the network in one pass.
num_steps - Number of characters in the sequence the network is trained on. Larger is better typically, the network will learn more long range dependencies. But it takes longer to train. 100 is typically a good number here.
lstm_size - The number of units in the hidden layers.
num_layers - Number of hidden LSTM layers to use
learning_rate - Learning rate for training
keep_prob - The dropout keep probability when training. If you're network is overfitting, try decreasing this.
"""


def download_dataset(filename,expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print("Found and Verified",filename)
    else:
        print(statinfo.st_size)
        raise Exception(
        "Failed to verify " + filename + ". Can you get it with a browser?"
        )
    return filename

def preprocessing(filename):
    with open(filename,"r") as f:
        lines = f.readlines()[46:1992]
        total_rows = len(lines)
        print("total rows:", len(lines))
        lines = [line for line in lines if not re.match(re.compile(r"[I]+\n"), line)]
        deletes_rows = total_rows - len(lines)
        print("deleted rows:", deletes_rows)
        lines = [line.replace("\n","").replace("  "," ") for line in lines]
        contents = "".join(lines)
        print("sample txt:",contents[:400])
    return contents

def get_word_dict(contents):
    vocab = sorted(set(contents))
    print("total vocab:",len(vocab))
    print("total words:",len(contents))
    vocab_to_int = {c:i for i,c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encoded = np.array([vocab_to_int[c] for c in contents], dtype=np.int32)
    return encoded , vocab, vocab_to_int, int_to_vocab

def get_bathes(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr) // characters_per_batch

    arr = arr[:characters_per_batch * n_seqs]
    arr = arr.reshape([n_seqs, characters_per_batch])

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:,n:n+n_steps]
        y = np.zeros_like(x)
        y[:,:-1] = x[:,1:]
        y[:,-1] = x[:,0]
        yield x,y

def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout
        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch
    '''
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name="inputs")
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name="targets")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.
        Arguments
        ---------
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size
    '''
    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits.
        Arguments
        ---------
        x: Input tensor
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer
    '''
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])

    with tf.variable_scope("softmax"):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size),stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x,softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    ''' Calculate the loss from the logits and the targets.
        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        lstm_size: Number of LSTM hidden units
        num_classes: Number of classes in targets
    '''
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.
        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer
    '''
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=128,
                    num_layers=2, learning_rate=0.001,
                    grad_clip=5, sampling=False):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()

        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # Build the LSTM cells
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, keep_prob)

        ## Run the data through RNN layers
        # First one-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs, num_classes)

        # Run each sequence step through the RNN and collect the outputs
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        # Get softmax predictions and logits
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        # Loss and Optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

def train(encoded, vocab):
    model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                    lstm_size=lstm_size, num_layers=num_layers,
                    learning_rate=learning_rate)
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess,"checkpoints/______.ckpt")
        counter = 0
        for e in range(epochs):
            new_state = sess.run(model.initial_state)
            loss = 0
            for x, y in get_bathes(encoded, batch_size, num_steps):
                counter += 1
                start = time.time()
                feed = {model.inputs:x,
                        model.targets:y,
                        model.keep_prob:keep_prob,
                        model.initial_state:new_state
                        }
                batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer],
                                                 feed_dict=feed)
                end = time.time()
                print('Epoch: {}/{}... '.format(e+1, epochs),
                      'Training Step: {}... '.format(counter),
                      'Training loss: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))

                if (counter % save_every_n == 0):
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

def pick_top_n(preds, vocab_size, top_n=1):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, vocab, vocab_to_int, int_to_vocab, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs:x,
                    model.keep_prob:1.0,
                    model.initial_state:new_state
                    }
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs:x,
                    model.keep_prob:1.0,
                    model.initial_state:new_state
                    }
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

    return "".join(samples)

def test(vocab, vocab_to_int, int_to_vocab):
    checkpoint = tf.train.latest_checkpoint('checkpoints')
    samp = sample(checkpoint, vocab, vocab_to_int, int_to_vocab, 1000, lstm_size, len(vocab), prime="The ")
    print(samp)

def main():
    filename = download_dataset(metamorphosis,141420)
    contents = preprocessing(metamorphosis)
    encoded , vocab, vocab_to_int, int_to_vocab = get_word_dict(contents)
    if False:
        train(encoded, vocab)
    else:
        test(vocab, vocab_to_int, int_to_vocab)

if __name__ == "__main__":
    main()
