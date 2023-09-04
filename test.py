# from gensim.test.utils import datapath, get_tmpfile
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np

# glove_file = datapath('/Users/miste/playground-rl/glove.42B.300d.txt')
# tmp_file = get_tmpfile("glove_word2vec.txt")

# _ = glove2word2vec(glove_file, tmp_file)

# model = KeyedVectors.load_word2vec_format("glove_word2vec.txt")

# print("Queen" in model)
embeddings = {}

words = []
with open("./wordlist.txt", 'r') as f:
      for line in f:
            words.append(line.strip())

add = []
with open("./top_100k.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in words:
                add.append(word)

with open("./top_100k.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in ('ice', 'cream', 'loch', 'ness', 'new', 'york', 'scuba', 'diver'):
                vector = np.asarray(values[1:], "float32")
                embeddings[word] = vector

def get_embed(word):
    first, second = word.split()
    first, second = np.array(embeddings[first]), np.array(embeddings[second])
    return np.add(first, second)/2

add = set(words) - set(add)
res = [get_embed(word) for word in add]

with open("extra.txt", 'w') as f:
    for w, vec in zip(add, res):
         vec = [str(i) for i in vec]
         line = f'{w} {" ".join(vec)}\n'
         f.write(line)
      
                  
# print(add)