import numpy as np
import matplotlib.pyplot as plt


rng =np.random.RandomState(1)
X =np.linspace(0,6,100)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
print(X)

fig = plt.figure()
#plt.plot(X)
plt.scatter(X,y)
plt.show()
