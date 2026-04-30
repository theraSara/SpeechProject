import torch
import torch.nn.functional as F

from classifiers.NCM import NearestClassMean
from models.utils import euclidean_dist


class OpenNCM(NearestClassMean):
    """
    Open Nearest Class Mean

    Extends NCM with an explicit unknown prototype. When '_unknown_' is already
    in class_list (via --speech.include_unknown), behaviour is identical to NCM.
    Alternatively, call fit_unknown() with K out-of-class samples to construct
    the unknown prototype separately, without enrolling it as a support class.
    """

    @torch.no_grad()
    def fit_unknown(self, unknown_x):
        """
        Build the unknown prototype from K out-of-class samples.
        Call this after fit_batch_offline() when '_unknown_' is not in class_list.

        :param unknown_x: tensor of shape [K, ...] or [1, K, ...] (raw waveforms or features)
        """
        if self.muK is None:
            raise RuntimeError("Call fit_batch_offline() before fit_unknown()")

        if self.cuda:
            unknown_x = unknown_x.cuda()

        unknown_x = unknown_x.view(-1, *unknown_x.size()[-unknown_x.dim()+1:]) if unknown_x.dim() > 2 \
                    else unknown_x
        z = self.backbone.get_embeddings(unknown_x)
        proto = z.mean(0, keepdim=True)  # [1, D]

        if '_unknown_' in self.word_to_index:
            # update the existing unknown prototype in-place
            idx = self.word_to_index['_unknown_']
            self.muK[idx] = proto.squeeze(0)
        else:
            self.muK = torch.cat([proto, self.muK], dim=0)
            self.class_list = ['_unknown_'] + list(self.class_list)
            self.word_to_index = {
                '_unknown_': 0,
                **{k: v + 1 for k, v in self.word_to_index.items()}
            }
            self.num_classes += 1