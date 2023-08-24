# from gensim.test.utils import datapath, get_tmpfile
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec

# glove_file = datapath('/Users/miste/playground-rl/glove.42B.300d.txt')
# tmp_file = get_tmpfile("glove_word2vec.txt")

# _ = glove2word2vec(glove_file, tmp_file)

# model = KeyedVectors.load_word2vec_format("glove_word2vec.txt")

# print("Queen" in model)

words = []
with open("./wordlist.txt", 'r') as f:
      for line in f:
            words.append(line.strip())

add = []
with open("./top_100k.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word == "smuggler" or word == "platypus":
                add.append(line)

with open("extra.txt", 'w') as f:
    for line in add:
         f.write(line)
      
                  
print(add)