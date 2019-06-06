import numpy as np
import matplotlib.pyplot as plt

path = ""
hist = np.loadtxt(path)

fig = plt.figure()
hist = hist.reshape(2, -1)
ax1 = fig.add_subplot(211)
ax1.plot(hist[0])

ax2 = fig.add_subplot(212)
ax2.plot(hist[1])

plt.show()

print(hist.shape)
