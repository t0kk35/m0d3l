"""
Helper Class for Embedding plotting.
(c) 2023 tsm
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from f3atur3s import FeatureCategorical

from ..common.exception import PlotException

from typing import Tuple

class EmbeddingPlot:
    @staticmethod
    def _val_dims(dims: int):
        if dims < 1 or dims > 3:
            raise PlotException(
                f'Dimensions parameter must be 1 or 2. Got <{dims}>'
            )

    @classmethod
    def decompose_and_plot(cls, feature: FeatureCategorical, embedding_weights: np.array, dims: int = 2,
                           fig_size: Tuple[float, float] = None):

        p = PCA(n_components=dims)
        p_e = p.fit_transform(embedding_weights)
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        if dims == 2:
            # Make 2 dim plot
            x, y = p_e[:, 0], p_e[:, 1]
            plt.scatter(x, y)
            # Annotate with Label
            for i, lb in sorted(list(feature.index_to_label.items()), key=lambda it: it[0]):
                plt.annotate(lb, (x[i], y[i]))
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
        elif dims == 3:
            # Make 3 dim plot
            x, y, z = p_e[:, 0], p_e[:, 1], p_e[:, 2]
            ax = plt.axes(projection='3d')
            ax.scatter3D(x, y, z)
            # Annotate with Label
            for i, lb in sorted(list(feature.index_to_label.items()), key=lambda it: it[0]):
                ax.text(x[i], y[i], z[i], lb)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        plt.title(f'Embedding {feature.name}. Explained variance {p.explained_variance_ratio_}')
        plt.show()
