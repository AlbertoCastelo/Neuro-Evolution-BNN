from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

COLORS = ['navy', 'turquoise', 'darkorange', 'forestgreen', 'red', 'darkblue', 'crimson', 'lawngreen', 'peru', 'grey']


def plot_dimensionality_reduction(x, y, index_to_plot, ax=None):
    pca = PCA(n_components=2)
    X_r = pca.fit(x).transform(x)

    n_classes = len(set(y))
    target_names = list(range(n_classes))
    colors = COLORS[:n_classes]
    lw = 0.5
    if not ax:
        fig, ax = plt.subplots()

    for color, i, target_name in zip(colors, target_names, target_names):
        ax.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                   label=target_name)
    ax.scatter(X_r[index_to_plot, 0], X_r[index_to_plot, 1], color='green', alpha=1.0, lw=3)
    ax.legend(loc='best', shadow=False, scatterpoints=1)
    ax.set_title('PCA')

    plt.show()
