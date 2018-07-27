import plac
import os
import numpy as np
from sklearn.neighbors import KDTree

ACTION_NN = 'nn'

class GloveEmbedding:
    def __init__(self, path):
        embedding_dim = int(os.path.basename(path).split('.')[2][:-1])
        self.load_glove_vectors(path, embedding_dim)
        self.tree = KDTree(self.embedding_matrix)

    def load_glove_vectors(self, path, embedding_dim):
        embeddings_index = {}
        f = open(path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        num_words = len(embeddings_index)
        embedding_matrix = np.zeros((num_words, embedding_dim))

        word2index = {}
        for index, (word, embedding) in enumerate(embeddings_index.items()):
            embedding_matrix[index] = embedding
            word2index[word] = index

        self.vocab = list(embeddings_index.keys())
        self.num_words = num_words
        self.embedding_matrix = embedding_matrix
        self.word2index = word2index

    def nearest(self, vector):
        dist, indexes = self.tree.query(vector, k=10)
        return dist, indexes

@plac.annotations(
    action=('What action to take.', 'option', 'a', str, [ACTION_NN]),
    glove_path=('Glove embeddings path', 'option', 'd', str),
    word=('Target word.', 'option', 'w', str))
def main(action, glove_path, word):
    embedding = GloveEmbedding(glove_path)
    if action == ACTION_NN:
        if word in embedding.word2index:
            word_idx = embedding.word2index[word]
            dist, indexes = embedding.nearest(embedding.embedding_matrix[word_idx].reshape(1, -1))
            for i, d in zip(indexes[0], dist[0]):
                w = embedding.vocab[i]
                print("word: {}, distance: {}".format(w, d))
        else:
            print('Word "{}" found'.format(word))

if __name__ == "__main__":
    plac.call(main)
