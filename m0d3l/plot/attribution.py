"""
Helper Class for captum plotting
(c) TSM 2023
"""
import matplotlib.pyplot as plt
import numpy as np

from ..common.attributionsresults import AttributionResultBinary
from ..common.exception import PlotException

from typing import Tuple

from f3atur3s import FeatureExpander, Feature, TensorDefinition
from f3atur3s import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_BINARY

class AttributionPlotBinary:
    def_style = 'ggplot'
    classification_colors = tuple([('FP', 'gold'), ('TP', 'red'), ('FN', 'orange'), ('TN', 'green')])

    @classmethod
    def overview(cls, attributions: AttributionResultBinary, fig_size: Tuple[float, float] = None):
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        plt.title('Average Attribution Per Feature')
        data_td = tuple(
            td for i, td in enumerate(attributions.tensor_definition) if i != attributions.original_label_index
        )
        nr_of_features = len([f for td in data_td for f in td.features])
        attrs = np.zeros((len(attributions), nr_of_features))
        f_names = []

        td_offset = 0
        for i, td in enumerate(data_td):
            f_offset = 0
            for j, f in enumerate(td.features):
                f_names.append(f.name)
                if isinstance(f, FeatureExpander) and f.learning_category == LEARNING_CATEGORY_BINARY:
                    cnt = len(f.expand_names)
                    attrs[0:, td_offset] = np.sum(attributions.attributions[i][0:, f_offset: f_offset + cnt], axis=1)
                    td_offset += 1
                    f_offset += cnt
                else:
                    attrs[0:, td_offset] = attributions.attributions[i][0:, f_offset]
                    td_offset += 1
                    f_offset += 1

        plt.violinplot(attrs)
        x_pos = (np.arange(1, len(f_names)+1))
        plt.xticks(x_pos, f_names, wrap=True)
        plt.xlabel('Feature')
        plt.ylabel('Attribution')
        plt.show()

    @classmethod
    def feature_detail(cls, attributions: AttributionResultBinary, f: Feature, fig_size: Tuple[float, float] = None):
        _ = plt.subplots(figsize=fig_size)
        plt.clf()
        plt.title(f'Attribution Detail for {f.name}')
        # Split out the various Tensor Definitions
        data_td = tuple(
            td for i, td in enumerate(attributions.tensor_definition) if i != attributions.original_label_index
        )

        # Find our feature
        td_i = [i for i, td in enumerate(data_td) if f in td.features]
        if len(td_i) > 1:
            raise PlotException(
                f'Found feature {f.name} in multiple tensor definitions ids {td_i}. Feature can only ' +
                f' be on one TensorDefinition'
            )
        else:
            td_i_a = td_i[0]
            td_i_d = attributions.original_x_indexes[td_i[0]]
        f_i = data_td[td_i_a].features.index(f)

        if f.learning_category == LEARNING_CATEGORY_CONTINUOUS:
            # Standard continuous feature, create a scatter plot of the original value vs the attribution.
            a = attributions.attributions[td_i_a][:, f_i]
            od = attributions.original_data[td_i_d][:, f_i]
            acl = attributions.classification_labels
            # Create plot
            for cl, c in cls.classification_colors:
                plt.scatter(od[np.where(acl == cl)], a[np.where(acl == cl)], c=c, label=cl, alpha=0.5)
            plt.legend()
            plt.xlabel(f.name)
            plt.ylabel('Attribution')
            plt.grid()
        elif f.learning_category == LEARNING_CATEGORY_BINARY and isinstance(f, FeatureExpander):
            # A one-hot feature. Create a vertical bar chart. The current should be all one hot features
            oh_i = (0,) + tuple(len(f.expand_names) for f in data_td[td_i_a].features)
            oh_i = (0,) + tuple(sr + st for sr, st in zip(oh_i, oh_i[1:]))
            a = attributions.attributions[td_i_a]
            a = a[:, oh_i[f_i]:oh_i[f_i+1]]
            acl = attributions.classification_labels
            bw = 0.20
            pos = np.arange(len(f.expand_names))
            for cl, c in cls.classification_colors:
                at = a[np.where(acl == cl)]
                if at.size != 0:
                    # Don't do anything if the slice as empty.
                    plt.barh(pos, at.mean(axis=0), bw, color=c, label=cl, alpha=0.8)
                    pos = [x + bw for x in pos]

            plt.xlabel('Average Attribution')
            plt.yticks([r + bw for r in range(len(f.expand_names))], [f.name for f in f.expand()])
            plt.legend()
            plt.grid(color='0.95')

        plt.show()

    @classmethod
    def heatmap(cls, attributions: AttributionResultBinary, td: TensorDefinition,
                fig_size: Tuple[float, float] = None, font_size=10):

        fig, ax = plt.subplots(figsize=fig_size)
        plt.title(f'Attribution heatmap', fontsize=20)

        # Make dictionary to look up colors
        cd = {cl: cr for cl, cr in cls.classification_colors}
        data_td = tuple(
            td for i, td in enumerate(attributions.tensor_definition) if i != attributions.original_label_index
        )

        # Find the TensorDefinition
        td_i = 0
        try:
            td_i = data_td.index(td)
        except ValueError:
            f'Can not find TensorDefinition {td.name} in input TensorDefinitions'

        a = attributions.attributions[td_i]
        od = attributions.original_data[td_i]
        acl = attributions.classification_labels

        im = ax.imshow(a, cmap='Blues', interpolation='nearest')

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                ax.text(j, i, f'{od[i, j].item():.2f}', ha='center', va='center', color=cd[acl[i]], fontsize=font_size)
        ax.set_ylabel('Sample number')
        tick_marks = np.arange(len(td.feature_names))
        ax.set_xticks(tick_marks, td.feature_names, rotation=45, rotation_mode="anchor",
                      ha="right")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Attribution', rotation=90)
        fig.tight_layout()

        plt.show()
