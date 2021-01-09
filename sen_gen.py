import numpy as np

corpus = ['red', 'blue', 'green', 'yellow', 'white', 'home', 'eat', 'black', 'food', 'what', 'when', 'shirt', 'bell', 'system', 'rain']
text = " ".join(list(np.random.choice(corpus, 3)))
print(text)
