import gensim
from collections import Counter
import math
import os
import random
import numpy as np
from six.moves import xrange
import tensorflow as tf
import pickle
import glob

path = "D:/doc2vec/data/MR"
categories = ['pos', 'neg']

model = gensim.models.Word2Vec.load("D:/word2vec/gensim/w2v_model/en_model_400.bin")

def read_data():
	files = []
	words_list = []
	for category in categories:
		file_list = glob.glob("%s/%s/*" % (path, category))
		for file in file_list:
			file_name = file.split("\\")[1]
			files.append(file_name)
			file_words = []
			for line in open(file, "r", encoding="utf-8"):
				file_words += line.split()
			words_list.append(file_words)
	return files, words_list

files, words_list = read_data()
print('Data size', len(files))
print('Data size', len(words_list))
print(files[0], words_list[0][:10])

vocabulary_size = 50000

def build_dataset(files, words_list, vocabulary_size):
	count = [['UNK', -1]]
	for file in files:
		count.append([file, 1])
	counter = Counter()
	for words in words_list:
		counter.update(words)
	count.extend(counter.most_common(vocabulary_size - 1))
	word2id = dict()
	for word, _ in count:
		word2id[word] = len(word2id)
	id2word = dict(zip(word2id.values(), word2id.keys()))
	data = list()
	unk_count = 0
	for words in words_list:
		file_data = []
		for word in words:
			if word in word2id:
				index = word2id[word]
			else:
				index = 0
				unk_count += 1
			file_data.append(index)
		count[0][1] = unk_count	
		data.append(file_data)
	return data, count, word2id, id2word

data, count, word2id, id2word = build_dataset(files, words_list, vocabulary_size)
print(len(word2id), len(id2word))
f = open("./output/word2id_pre_train.dm", "wb")
pickle.dump(word2id, f)
f.close()

f = open("./output/id2word_pre_train.dm", "wb")
pickle.dump(id2word, f)
f.close()

print('Most common words (+UNK)', count[:5])
print('Sample data', data[0][:10], [id2word[i] for i in data[0][:10]])


def generate_batch(batch_size, window_size):
	batch = np.ndarray(shape=(batch_size, window_size+1), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	for i in range(batch_size):
		file = random.choice(files)
		file_id = word2id[file]
		file_index = files.index(file)
		words = words_list[file_index]
		random_index = random.randrange(0, len(words) - window_size)
		words = words[random_index : random_index + window_size + 1]
		word_batch = []
		for word in words:
			if word in word2id:
				word_batch.append(word2id[word])
			else:
				word_batch.append(0)
		batch[i] = [file_id] + word_batch[:window_size]
		labels[i, 0] = word_batch[window_size]
	return batch, labels

batch, labels = generate_batch(batch_size=8, window_size=3)
for i in range(8):
	print(batch[i, 0], id2word[batch[i, 0]],
			batch[i, 1], id2word[batch[i, 1]],
			batch[i, 2], id2word[batch[i, 2]],
			batch[i, 2], id2word[batch[i, 3]],
			'->', labels[i, 0], id2word[labels[i, 0]])


window_size = 8
batch_size = 128
embedding_size = 400
vocab_size = len(word2id)
num_sampled = 64

def embedding(vocab_size, embedding_size):
	w = np.random.rand(vocab_size, embedding_size)
	for id in id2word:
		word = id2word[id]
		if 'txt' not in word:
			if word in model:
				w[id] = model[word]
	return w

graph = tf.Graph()

with graph.as_default():
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size, window_size+1])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

	embeddings = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=True, name="embeddings")
	embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])

	with tf.device('/cpu:0'):	
		embedding_init = embeddings.assign(embedding_placeholder)
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)
		embed_context = tf.reduce_mean(embed, 1)

		nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([vocab_size]))

	loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed_context, num_sampled, vocab_size))

	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm

	init = tf.global_variables_initializer()

num_steps = 1000001

with tf.Session(graph=graph) as session:
	init.run()
	print("Initialized")

	average_loss = 0
	for step in xrange(num_steps):
		batch_inputs, batch_labels = generate_batch(batch_size, window_size)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
			print("Average loss at step ", step, ": ", average_loss)
			average_loss = 0

	final_embeddings = normalized_embeddings.eval()
	f = open("./output/embeddings_pre_train.dm", "wb")
	pickle.dump(final_embeddings, f)
	f.close()

