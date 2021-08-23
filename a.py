import numpy as np
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(open('1231_自然数.csv'))
labels = np.array(data.columns.values)

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=0)
pos = mds.fit_transform(data)
res = pd.DataFrame(pos, columns=['x', 'y'])
plt.scatter(res['x'], res['y'], marker='.')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

for label, x, y in zip(labels, pos[:, 0], pos[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(50, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )

plt.show()
