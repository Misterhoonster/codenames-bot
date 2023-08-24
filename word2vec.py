# from gensim.models import KeyedVectors
# import gensim.downloader
import numpy as np
from scipy import spatial
from itertools import combinations
from nltk.corpus import wordnet
import nltk
import numpy as np

try:
    wordnet.ensure_loaded()
except:
    nltk.download("wordnet")

word_corpus = set(wordnet.words())

# finds distances from every possible hint to each square
def compute_distances(targets, opps, embeddings):
    d = {}
    hints = embeddings.keys()
    for hint in hints:
        for target in targets:
            d[(hint, target)] = distance(hint, target, embeddings)
        for opp in opps:
            d[(hint, opp)] = distance(hint, opp, embeddings)
    return d

# finds distance between single hint and target word
def distance(hint, target, embeddings):
    try:
        return spatial.distance.cosine(embeddings[hint], embeddings[target])
    except:
        first, second = target.split()
        first, second = np.array(embeddings[first]), np.array(embeddings[second])
        target_embed = np.add(first, second)/2
        return spatial.distance.cosine(embeddings[hint], target_embed)

# sums the distances from a hint to each target word
def accuracy(hint, targets, opps, d):
    return sum([d[(hint, opp)] for opp in opps]) - sum([d[(hint, target)] for target in targets])

# minimax from the hint to target and opponent words
def precision(hint, targets, opps, d):
    val = min([d[(hint, opp)] for opp in opps]) - max([d[(hint, target)] for target in targets])
    return val if val > 0 else 0

# computes score based on accuracy * precision
def score(hint, targets, opps, d):
    return accuracy(hint, targets, d) * precision(hint, targets, opps, d)

def is_valid_word(word, board_words):
    for board_word in board_words:
        if word in board_word or board_word in word:
            return False
    return True

# returns the highest scoring hint
def ranking(targets, team_squares, opps, embeddings, d):
    board_words = team_squares + opps

    top_acc = sorted(embeddings.keys(), key = lambda w: -1 * accuracy(w, targets, opps, d))[:250]
    top_prec = sorted(top_acc, key = lambda w: -1 * precision(w, targets, opps, d))[:100]

    scores = zip([precision(w, targets, opps, d) for w in top_prec], top_prec)
    filtered_scores = list(filter(lambda w: is_valid_word(w[1], board_words), scores))
    return max(filtered_scores)
    # ranked = sorted(embeddings.keys(), key = lambda w: -1 * score(w, targets, opps, d))
    # return [f'{i+1}. {w} {score(w, targets, opps, d):.4f}' for i, w in enumerate(ranked[:50])]

# get embeddings for top 50k words
def load_embeddings(*board):
    embeddings = {}
    if board:
        board = board[0]
    with open("./top_50000.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_corpus or (board and word in board):
                vector = np.asarray(values[1:], "float32")
                embeddings[word] = vector
    return embeddings

# generate best guess based on wv similarities
def generate_guess(open_squares, clue, count):
    embeddings = load_embeddings()

    d = [distance(w, clue, embeddings) for _, w in open_squares]
    top = sorted(zip(open_squares, d), key= lambda x: x[1])[:count]
    return [i for (i, _), _ in top]

# generates best hint based on word vector similarities
def generate_clue(team_squares, opp_squares):
    board = team_squares + opp_squares
    embeddings = load_embeddings(board)
    
    # compute distances from all possible hints to each word on the board
    d = compute_distances(team_squares, opp_squares, embeddings)

    # generate all possible combos of target words from size 2 to all
    max_len = min(len(team_squares), 4)
    targets = [list(combinations(team_squares, n)) for n in range(1, max_len+1)]
    scores = [0]*len(targets)

    # loop through all combos finding the best clue for each combo size
    for l, arr in enumerate(targets):
        best = [0, "n/a"]
        for target in arr:
            cur_best = ranking(target, team_squares, opp_squares, embeddings, d)
            if cur_best[0] > best[0]:
                best = cur_best
        best = [np.log(l+1)*best[0], best[1]]
        scores[l] = best
    
    # return the word and count for the best hint
    res = ["Clue", 1]
    max_s = 0
    for l, (s, w) in enumerate(scores):
        if s > max_s:
            res = [w, l+1]
            max_s = s
    print(res)
    return res

if __name__ == "__main__":
    team_squares = ["iron", "ham", "beijing", "superhero"]
    opp_squares = ["witch", "fall", "note", "cat", "bear", "ambulance"]
    ipt = [(i, w) for i, w in enumerate(team_squares+opp_squares)]
    # generate_guess(ipt, "wok", 3)
    generate_clue(team_squares, opp_squares)
