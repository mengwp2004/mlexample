#encoding=utf-8

from minepy import MINE
import numpy as np

m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x**2)
print( m.mic())
