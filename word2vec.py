import numpy as np
from scipy import spatial
from itertools import combinations
from nltk.corpus import wordnet
import nltk
import numpy as np
import heapq

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
        print('word not found :(', hint, target)
        return 1
        # first, second = target.split()
        # first, second = np.array(embeddings[first]), np.array(embeddings[second])
        # target_embed = np.add(first, second)/2
        # return spatial.distance.cosine(embeddings[hint], target_embed)

# sums the distances from a hint to each target word
def accuracy(hint, targets, opps, d):
    return np.mean([d[(hint, opp)] for opp in opps]) - np.mean([d[(hint, target)] for target in targets])
    return sum([d[(hint, opp)] for opp in opps]) - sum([d[(hint, target)] for target in targets])

# minimax from the hint to target and opponent words
def precision(hint, targets, opps, d):
    val = min([d[(hint, opp)] for opp in opps]) - max([d[(hint, target)] for target in targets])
    return val if val > 0 else 0

# computes score based on accuracy * precision
def score(hint, targets, opps, d):
    return accuracy(hint, targets, d) * precision(hint, targets, opps, d)

# check if clue meets the game reqs
def is_valid_word(word, targets, opps, d, board_words):
    if not word.isalpha() or " " in word:
        return False
    if precision(word, targets, opps, d) == 0:
        return False
    for board_word in board_words:
        if word in board_word or board_word in word:
            return False
    return True

# returns the highest scoring hint
def ranking(targets, team_squares, opps, embeddings, d):
    board_words = team_squares + opps

    top_acc = sorted(embeddings.keys(), key = lambda w: -1 * accuracy(w, targets, opps, d))[:250]
    scores = zip([accuracy(w, targets, opps, d) for w in top_acc], top_acc)
    filtered_scores = list(filter(lambda w: is_valid_word(w[1], targets, opps, d, board_words), scores))
    return max(filtered_scores)


    # top_prec = sorted(top_acc, key = lambda w: -1 * precision(w, targets, opps, d))
    # scores = zip([precision(w, targets, opps, d) for w in top_prec], top_prec)
    # filtered_scores = list(filter(lambda w: is_valid_word(w[1], board_words), scores))
    # return max(filtered_scores)

    top_acc = sorted(embeddings.keys(), key = lambda w: -1 * accuracy(w, targets, opps, d))[:250]
    top_prec = sorted(top_acc, key = lambda w: -1 * precision(w, targets, opps, d))
    scores = zip([precision(w, targets, opps, d) for w in top_prec], top_prec)
    filtered_scores = list(filter(lambda w: is_valid_word(w[1], board_words), scores))
    return max(filtered_scores)

# get embeddings for top 50k words
def load_embeddings(board):
    embeddings = {}
    # add all words in corpus + board to embed
    with open("./top_50000.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_corpus or word in board:
                vector = np.asarray(values[1:], "float32")
                embeddings[word] = vector
    # handle word pairs
    with open("./word_pairs.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0] + " " + values[1]
            vector = np.asarray(values[2:], "float32")
            embeddings[word] = vector
    return embeddings

# generate best guess based on wv similarities
def generate_guess(open_squares, clue, count):
    embeddings = load_embeddings([w for _, w in open_squares])
    d = [distance(w, clue, embeddings) for _, w in open_squares]
    top = sorted(zip(open_squares, d), key= lambda x: x[1])[:count]
    guesses = [i for (i, _), d in top if d >= 0.3]
    print(clue, [(w, round(d, 2)) for (_, w), d in top])
    guesses = guesses + [-1] if len(guesses) < count else guesses
    return guesses

def get_k_best_clues(team_squares, opp_squares, embeddings, d, k=5):
    # generate all possible combos of target words from size 2 to all
    max_len = min(len(team_squares), 4)
    targets = [list(combinations(team_squares, n)) for n in range(1, max_len+1)]
    scores = [0]*len(targets)

    # NEW VERSION
    scores = []

    # loop through all combos finding the best clue for each combo size
    for l, arr in enumerate(targets):
        for target in arr:
            score, word = ranking(target, team_squares, opp_squares, embeddings, d)
            # score /= (l+1)
            score *= (l+1)**(len(team_squares)/6.0)
            heapq.heappush(scores, (-score, (word, target)))
    # get k best
    best = heapq.nsmallest(k, scores)
    best = [(key, -value) for value, key in best]
    return best

# generates best hint based on word vector similarities
def generate_clue(team_squares, opp_squares):
    board = team_squares + opp_squares
    embeddings = load_embeddings(board)
    
    # compute distances from all possible hints to each word on the board
    d = compute_distances(team_squares, opp_squares, embeddings)

    best = get_k_best_clues(team_squares, opp_squares, embeddings, d)

    res = ["Clue", 1]
    top_score = 0

    for (word, target), score in best:
        updated_words = list(set(team_squares) - set(target))
        if updated_words:
            _, nxt_score = get_k_best_clues(updated_words, opp_squares, embeddings, d, k=1)[0]
            score += nxt_score
            score /= 2
        if score > top_score:
            top_score = score
            res = [word, len(target)]
    print(res)
    return res

if __name__ == "__main__":
    team_squares = ["iron", "pig", "beijing"]
    opp_squares = ["witch", "fall", "note", "cat", "bear", "ambulance"]
    ipt = [(i, w) for i, w in enumerate(team_squares+opp_squares)]
    generate_guess(ipt, "wok", 3)
    generate_clue(team_squares, opp_squares)
