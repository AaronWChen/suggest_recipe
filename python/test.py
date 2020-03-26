import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import collections
import numpy as np

train_path = "../raw_data/train.json"
save_path = "../write_data"
embedding_size = 100
epochs_desired = 15
learning_rate = 0.025
regularization = 0.01
algo_optimizer = 'adam'

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
  print('Sample data', data[0][:10], 
        [reverse_dictionary[i] for i in data[0][:10]])
  global sentence_index
  global words_processed
  sentence_index = 0
  words_processed = 0
  print('example batch: ')
  batch, labels = generate_batch(data, words_per_epoch, count)
  for i in range(len(batch)):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', [w for w in labels[i]], 
          [reverse_dictionary[w] for w in labels[i]])
  valid_size = 16     # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick words in the head of the distribution
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)
  
  train_inputs = tf.Variable(initial_value=tf.ones([1,1],
                                                   dtype=tf.int32),
                             validate_shape=False)

  train_labels = tf.Variable(initial_value=tf.ones([1,1],
                                                   dtype=tf.int32),
                             validate_shape=False)

  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  words_processed_ph = tf.Variable(initial_value=tf.zeros(
                                                      [1,1], 
                                                      dtype=tf.int32), 
                                    validate_shape=False)
  words_to_train = float(words_per_epoch * epochs_desired)

  lr = learning_rate * tf.maximum(0.0001, 
                                  1.0 - tf.cast(words_processed_ph, 
                                                tf.float32) / words_to_train)

  model = keras.Sequential([
                            keras.layers.Embedding(input_dim=vocabulary_size, 
                                                   output_dim=embedding_size, 
                                                  ),
                            keras.layers.GlobalAveragePooling1D(),
                            keras.layers.Dense(1, activation='softmax')
                          ])

  if algo_optimizer == 'sgd':
    model.compile(optimizer=keras.optimizers.SGD(lr), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
  elif algo_optimizer == 'adam':
    model.compile(optimizer=keras.optimizers.Adam(lr, 
                                                  beta_1=0.9, 
                                                  beta_2=0.999, 
                                                  epsilon=1e-6), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])



  num_steps = 1000001

  average_loss = 0.
  sentences_to_train = epochs_desired * len(data)

  for step in range(num_steps):
    if step < sentences_to_train:
      batch_inputs, batch_labels = generate_batch(train_data, 
                                                  words_per_epoch, 
                                                  count)

      #feed_dict = {train_inputs: batch_inputs,
      #              train_labels: batch_labels,
      #              words_processed_ph.experimental_ref(): words_processed}



  #print(model.summary())
  print(train_data)
  #model.fit(np.array(train_data), epochs=epochs_desired)  
  #model.evaluate(np.array(vad_data))
train()