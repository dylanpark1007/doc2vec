from  collections import Counter
import math
import os
import random
import glob
import numpy as np
import tensorflow as tf
import pickle
from six.moves import xrange 

path = "D:/doc2vec/data/MR"
categories = ['pos', 'neg']

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
f = open("./output/word2id2.bin", "wb")
pickle.dump(word2id, f)
f.close()

f = open("./output/id2word2.bin", "wb")
pickle.dump(id2word, f)
f.close()

print('Most common words (+UNK)', count[:5])
print('Sample data', data[0][:10], [id2word[i] for i in data[0][:10]])

def generate_batch(batch_size):
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	for i in range(batch_size):
		file = random.choice(files)
		batch_index = word2id[file]
		file_index = files.index(file)
		word = random.choice(words_list[file_index])
		if word in word2id:
			label = word2id[word]
		else:
			label = 0
		batch[i] = batch_index
		labels[i] = label
	return batch, labels

batch, labels = generate_batch(batch_size=8)
for i in range(8):
  print(batch[i], id2word[batch[i]],
        '->', labels[i, 0], id2word[labels[i, 0]])

batch_size = 128
vocab_size = len(word2id)
embedding_size = 400
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

	with tf.device('/cpu:0'):
		embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)
		nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([vocab_size]))
		print(embeddings)
		print(embed)
		print(nce_weights)
		print(nce_biases)
		print(num_sampled)
		print(vocab_size)

	loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
										biases=nce_biases,
										labels=train_labels,
										inputs=embed,
										num_sampled=num_sampled,
										num_classes=vocab_size))

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
		batch_inputs, batch_labels = generate_batch(batch_size)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val
		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
			print("Average loss at step ", step, ": ", average_loss)
			average_loss = 0

	final_embeddings = normalized_embeddings.eval()
	f = open("./output/embeddings2.bin", "wb")
	pickle.dump(final_embeddings, f)
	f.close()
