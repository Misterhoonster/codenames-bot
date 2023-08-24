import numpy as np

with open("./top_50000.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word == "platypus": print("FOUND")
print('done')