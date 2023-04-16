"""
Class that helps bridge a m0d3l and captum.
"""
import numpy as np
import torch

from ..pytorch.models.base import ModelTensorDefinition
from ..common.attributionsresults import AttributionResultBinary
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

from typing import Tuple, Type

class CaptumAttributions:
    @classmethod
    def get_attributions_binary(cls, model: ModelTensorDefinition, device: torch.device, captum_cls: Type,
                                sample: Tuple[torch.Tensor]) -> AttributionResultBinary:
        inp = model.get_x(sample)
        # Get the classification result, we'll use this to assign the FP, TP, FN and TN labels.
        model.eval()
        model.to(device)
        with torch.no_grad():
            inp = tuple(i.to(device, non_blocking=True) for i in inp)
            y_prd = model(inp)[0]
        y_prd = np.squeeze((y_prd > 0.5).detach().cpu().numpy())
        y = np.squeeze(model.get_y(sample)[0].detach().cpu().numpy())
        class_res = np.empty(y.shape[0], dtype=np.object)
        for v_y, v_prd, label in [(1, 1, 'TP'), (0, 1, 'FP'), (0, 0, 'TN'), (1, 0, 'FN')]:
            class_res[np.where((y == v_y) & (y_prd == v_prd))] = label

        # TODO this will only work if the label is the last tensor definition. Need to get data tds and use indexes
        # Re-order the input a bit, captum wants interpretable embeddings, not the original input for categorical.
        inp_w_emb = []
        i_emb = []
        for i, (td, o_inp) in enumerate(zip(model.tensor_definitions, inp)):
            if len(td.categorical_features()) > 0:
                # All features should be categorical
                emb = []
                for j, f in enumerate(td.features):
                    layer_name = 'heads.' + str(i) + '.embedding.embeddings.' + str(j)
                    iel = configure_interpretable_embedding_layer(model, layer_name)
                    i_emb.append(iel)
                    emb.append(iel.indices_to_embeddings(o_inp[:, j]))
                emb = torch.cat(emb, dim=1)
                inp_w_emb.append(emb)
            else:
                inp_w_emb.append(inp[i])

        inp_w_emb = tuple(inp_w_emb)

        # Set the TensorDefinitionHead Layers in CaptumMode
        for tdh in model.tensor_definition_heads:
            tdh.captum_mode = True

        # Get attributions from captum
        ctm = captum_cls(model.forward_captum)
        attr = ctm.attribute(inp_w_emb, target=0, return_convergence_delta=False)

        attr_new = []
        # Sum the attributions for embeddings, the dimensions do not make sense.
        for i, (h, at) in enumerate(zip(model.tensor_definition_heads, attr)):
            if h.embedding is None:
                attr_new.append(at)
            else:
                off_set = 0
                emb_a = []
                for e in h.embedding.embeddings:
                    size = e.embedding_dim
                    x = attr[i][0:, off_set:size+off_set]
                    x = torch.unsqueeze(torch.sum(x, dim=-1), dim=1)
                    emb_a.append(x)
                    off_set += size
                emb_a = torch.cat(emb_a, dim=1)
                attr_new.append(emb_a)

        attr = tuple(a.cpu().detach().numpy() for a in attr_new)

        # Remove interpretable embeddings
        for iel in i_emb:
            remove_interpretable_embedding_layer(model, iel)

        # Set the TensorDefinitionHead Layers in NONE CaptumMode
        for tdh in model.tensor_definition_heads:
            tdh.captum_mode = False

        # Get the original input in as numpy
        orig_inp = tuple(s.cpu().detach().numpy() for s in sample)
        tds = tuple(td for i, td in enumerate(model.tensor_definitions))
        return AttributionResultBinary(tds, orig_inp, model.x_indexes, model.label_index, class_res, attr)
