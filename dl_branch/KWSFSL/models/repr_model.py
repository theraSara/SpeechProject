import torch.nn as nn
import torch.nn.functional as F


from models.utils import register_model
from models.encoder.DSCNN import DSCNNL, DSCNNM, DSCNNS, DSCNNL_NONORM, DSCNNL_LAYERNORM, DSCNNS_NONORM, DSCNNS_LAYERNORM


from models.preprocessing import MFCC

from models.losses.triplet import online_triplet_loss



class ReprModel(nn.Module):
    def __init__(self, encoder, preprocessing, criterion, x_dim, emb_norm):
        super(ReprModel, self).__init__()
        self.encoder = encoder
        self.preprocessing = preprocessing
        self.emb_norm = emb_norm

        if criterion['type'] == 'triplet':
            self.criterion = online_triplet_loss(criterion)

    def get_embeddings(self, x):
        if self.preprocessing:
            x = self.preprocessing.extract_features(x)
        zq = self.encoder.forward(x)
        if self.emb_norm:
            zq = F.normalize(zq, p=2.0, dim=-1)
        return zq

    def loss(self, x):
        n_class = x.size(0)
        n_sample = x.size(1)

        x = x.view(n_class * n_sample, *x.size()[2:]).cuda()
        zq = self.get_embeddings(x)

        loss_val = self.criterion.compute(zq, n_sample, n_class)

        return loss_val, {
            'loss': loss_val.item(),
        }

    def loss_class(self, x, labels):
        zq = self.get_embeddings(x)
        return self.criterion.compute(zq, labels)



def get_encoder(encoding, x_dim, hid_dim, out_dim):
    if encoding == 'DSCNNL':
        return DSCNNL(x_dim)
    elif encoding == 'DSCNNL_NONORM':
        return DSCNNL_NONORM(x_dim)
    elif encoding == 'DSCNNL_LAYERNORM':
        return DSCNNL_LAYERNORM(x_dim)
    elif encoding == 'DSCNNM':
        return DSCNNM(x_dim)
    elif encoding == 'DSCNNS':
        return DSCNNS(x_dim)
    elif encoding == 'DSCNNS_NONORM':
        return DSCNNS_NONORM(x_dim)
    elif encoding == 'DSCNNS_LAYERNORM':
        return DSCNNS_LAYERNORM(x_dim)
    else:
        raise ValueError("Model {} is not valid".format(encoding))


@register_model('repr_conv')
def load_repr_conv(**kwargs):
    z_norm = kwargs['z_norm']
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    encoding = kwargs['encoding']
    print(encoding, x_dim, hid_dim, z_dim)

    encoder = get_encoder(encoding, x_dim, hid_dim, z_dim)

    preprocessing = False
    if 'mfcc' in kwargs.keys():
        audio_prep = kwargs['mfcc']
        preprocessing = MFCC(audio_prep)

    criterion = kwargs['loss'] if 'loss' in kwargs.keys() else False

    return ReprModel(encoder, preprocessing, criterion, x_dim, z_norm)
