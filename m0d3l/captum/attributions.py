"""
Class that helps bridge a m0d3l and captum.
"""
import numpy as np
import torch

from ..pytorch.models.base import ModelTensorDefinition
from ..common.attributionsresults import AttributionResultBinary

from typing import Tuple, Type

class CaptumAttributions:
    @classmethod
    def get_attributions_binary(cls, model: ModelTensorDefinition, captum_cls: Type,
                                sample: Tuple[torch.Tensor]) -> AttributionResultBinary:
        inp = model.get_x(sample)
        # Get the classification result
        model.eval()
        model.to(torch.device('cpu'))
        with torch.no_grad():
            y_prd = model(inp)[0]
        y_prd = np.squeeze((y_prd > 0.5).detach().cpu().numpy())
        y = np.squeeze(model.get_y(sample)[0].detach().cpu().numpy())
        class_res = np.empty(y.shape[0], dtype=np.object)
        for v_y, v_prd, label in [(1, 1, 'TP'), (0, 1, 'FP'), (0, 0, 'TN'), (1, 0, 'FN')]:
            class_res[np.where((y == v_y) & (y_prd == v_prd))] = label
        # Get attributions from captum
        ctm = captum_cls(model.forward_captum)
        attr = ctm.attribute(inp, target=0, return_convergence_delta=False)
        attr = tuple(a.detach().cpu().numpy() for a in attr)
        orig_inp = tuple(s.detach().cpu().numpy() for s in sample)
        tds = tuple(td for i, td in enumerate(model.tensor_definitions))
        return AttributionResultBinary(tds, orig_inp, model.x_indexes, model.label_index, class_res, attr)
