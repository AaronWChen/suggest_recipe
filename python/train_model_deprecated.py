""" Train a multi-class classification problem.

Use the embedding for each ingredient in a recipe to suggest and predict other
recipes a user will like. They suggested recipes should not be in the same 
cuisine category.

Debug the model with a held-out validation set.

This model is heavily based on the work of Jaan Altosaar: github.com/altosaar
and the project food2vec: github.com/altosaar/food2vec.

The data for this version of the project, as well as the forked version of 
Altosar's work, is provded by Yummly from the Kaggle competition "What's 
Cooking?": https://www.kaggle.com/c/whats-cooking

Future data sources will be cited here.
"""

# Set initial parameters and import necessary modules
import tensorflow as tf
import tensorflow_addons as tfa
import collections
import numpy as np

#layers = tfa.contrib.layers

train_path = "../raw_data/train.json"
save_path = "../write_data"
embedding_size = 100
epochs_desired = 15
learning_rate = 0.025
regularization = 0.01
algo_optimizer = 'adam'

"""# Functions used to handle complex strings
def complex_flatten(list_of_lists):
  for x in list_of_lists:
    if hasattr(x, '__iter__') and not isinstance(x, str):
      for y in complex_flatten(x):
        yield y

    else:
      yield x

def dummy_fun(doc):
  return doc

def tokenizer(doc):
  for sen in doc:
    tokens = [word_tokenize(word) for word in sen]
    flat_tkns = list(complex_flatten(tokens))
    stopped_tokens = [w.lower() for w in flak_tkns if w not in stopwords_list]
    token_recipes.append(lemma_tokens)"""

# Functions used from food2vec to import and process data
flatten = lambda l: [item for sublist in l for item in sublist]

def read_data(train_path):
  sentences = []
  with open(train_path, 'rb') as f:
    for line in f:
      sentences.append(line.rstrip().split())
  return sentences


def build_dataset(sentences, min_count=0):
  count = [['UNK', -1]]
  sentences_flat = flatten(sentences)
  counter = collections.Counter(sentences_flat)
  n = len(counter)
  filt = [(word, c) for word, c in counter.most_common(n) if c > min_count]
  count.extend(filt)
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for sentence in sentences:
    sentence_ids = []
    for word in sentence:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count += 1
      sentence_ids.append(index)
    data.append(sentence_ids)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary


def build_train_validation(data, validation_fraction=0.1):
  vad_idx = np.random.choice(
      range(len(data)), int(validation_fraction * len(data)), replace=False)
  raw_vad_data = [data[i] for i in vad_idx]
  train_data = [data[i] for i in list(set(range(len(data))) - set(vad_idx))]
  train_counts = collections.Counter(flatten(train_data))
  vad_data = []
  for vad_sentence in raw_vad_data:
    if any(word not in train_counts for word in vad_sentence):
      train_data.append(vad_sentence)
    else:
      vad_data.append(vad_sentence)
  print(f"""Split data into {len(train_data)} train and {len(vad_data)} 
        validation""")
  return train_data, vad_data


def generate_batch(data, corpus_size, count, subsample=1e-3):
  global sentence_index
  global words_processed
  raw_sentence = data[sentence_index]
  if subsample == 0.:
    sentence = raw_sentence
  else:
    sentence = []
    for word_id in raw_sentence:
      word_freq = count[word_id][1]
      keep_prob = ((np.sqrt(word_freq / (subsample * corpus_size)) + 1) *
                   (subsample * corpus_size) / word_freq)
      if np.random.rand() > keep_prob:
        pass
      else:
        sentence.append(word_id)
    if len(sentence) < 2:
      sentence = raw_sentence
  sentence_index = (sentence_index + 1) % len(data)
  return get_sentence_inputs(sentence, len(count))


def get_sentence_inputs(sentence, vocabulary_size):
  sentence_set = set(sentence)
  batch = np.asarray(sentence, dtype=np.int32)
  labels = np.asarray(
      [list(sentence_set - set([w])) for w in sentence], dtype=np.int32)
  return batch, labels


def train():
  raw_data = read_data(train_path)
  data, count, dictionary, reverse_dictionary = build_dataset(raw_data)
  train_data, vad_data = build_train_validation(data)
  vocabulary_size = len(dictionary)
  words_per_epoch = len(flatten(train_data))
  sentences_per_epoch = len(train_data)
  del raw_data # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[0][:10], [reverse_dictionary[i] for i in data[0][:10]])
  global sentence_index
  global words_processed
  sentence_index = 0
  words_processed = 0
  print('example batch: ')
  batch, labels = generate_batch(data, words_per_epoch, count)
  for i in range(len(batch)):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', [w for w in labels[i]], [reverse_dictionary[w] for w in labels[i]])
  valid_size = 16     # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick words in the head of the distribution
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)

  graph = tf.Graph()
  with graph.as_default():
    # Input data.
    #train_inputs = tf.placeholder(tf.int32, shape=[None])
    train_inputs = tf.Variable(initial_value=tf.zeros([1,1], 
                                                      dtype=tf.int32), 
                                validate_shape=False,
                                dtype=tf.int32)
    
    #train_labels = tf.placeholder(tf.int32, shape=[None, None])
    train_labels = tf.Variable(initial_value=tf.zeros([1,1], 
                                                      dtype=tf.int32), 
                                validate_shape=False,
                                dtype=tf.int32
                                )
    train_indicators = tf.one_hot(
          train_labels, depth=vocabulary_size, on_value=1, off_value=0, axis=1)
    print(train_indicators)
    
    #train_indicators = tf.to_float(tf.reduce_sum(train_indicators, -1))
    train_indicators = tf.dtypes.cast(tf.reduce_sum(train_indicators, -1),
                                          dtype=tf.float32)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      embeddings = tf.Variable(
          tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      example_emb = tf.nn.embedding_lookup(embeddings, train_inputs)
      # Construct the variables for the softmaxloss
      sm_w_t = tf.Variable(
          tf.zeros([vocabulary_size, embedding_size]))
      sm_b = tf.Variable(tf.zeros([vocabulary_size]))
    # logits: [batch_size, vocab_size]
    logits = tf.matmul(example_emb, sm_w_t, transpose_b=True) + sm_b
    print(logits.shape)
    # Compute the average loss for the batch.
    log_lik = tf.reduce_mean(-tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.reshape(logits, (1, -1)), labels=train_indicators))
    regularizer_loss = (regularization * (
      tf.nn.l2_loss(sm_w_t) + tf.nn.l2_loss(example_emb)))
    loss = tf.reduce_mean(-log_lik) + regularizer_loss

    # Construct the SGD optimizer using a decaying learning rate
    #words_processed_ph = tf.placeholder(tf.int32, [])
    words_processed_ph = tf.Variable(initial_value=tf.zeros([1,1], 
                                                      dtype=tf.int32), 
                                      validate_shape=False)
    words_to_train = float(words_per_epoch * epochs_desired)
    lr = learning_rate * tf.maximum(
        0.0001, 1.0 - tf.cast(words_processed_ph, tf.float32) / words_to_train)

    if algo_optimizer == 'sgd':
      optimizer - tf.keras.optimizers.SGD(lr)
      #optimizer = tf.keras.train.GradientDescentOptimizer(lr)
    elif algo_optimizer == 'adam':
      optimizer = tf.keras.optimizers.Adam(lr,
                                            beta_1=0.9, 
                                            beta_2=0.999, 
                                            epsilon=1e-6)
      #optimizer = tf.compat.v1.train.AdamOptimizer(
      #    lr, beta1=0.9, beta2=0.999, epsilon=1e-6)
    train_op = optimizer.minimize(loss, var_list=embeddings)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

    num_steps = 1000001

  with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()

    average_loss = 0.
    sentences_to_train = epochs_desired * len(data)
    for step in range(num_steps):
      if step < sentences_to_train:
        batch_inputs, batch_labels = generate_batch(train_data, words_per_epoch, count)
        feed_dict = {train_inputs: batch_inputs,
                     train_labels: batch_labels,
                     words_processed_ph: words_processed}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([train_op, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # average loss is an estimate of the loss over the last 2000 batches.
          print("Average train loss at step %d: %.3e" % (step, average_loss))
          average_loss = 0.

        # Calculate held-out log-likelihood once per epoch
        if step % sentences_per_epoch == 0 and step > 0:
          vad_log_lik = 0.
          for sentence in vad_data:
            batch_inputs, batch_labels = get_sentence_inputs(
                sentence, vocabulary_size)
            log_lik_val = session.run(
                log_lik,
                {train_inputs: batch_inputs, train_labels: batch_labels})
            vad_log_lik += np.sum(log_lik_val)
          print("Average validation log-likelihood at step %d: %.3e" %
              (step, vad_log_lik / len(vad_data)))

        # Note that this is expensive
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = "%s %s," % (log_str, close_word)
            print(log_str)
    final_embeddings = normalized_embeddings.eval()

def main(_):
  train()

if __name__ == '__main__':
  tf.compat.v1.app.run()
