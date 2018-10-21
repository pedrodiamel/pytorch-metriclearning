import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbHingeLoss(nn.Module):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3 and a margin with a value greater than 0.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    .. math::
        L(a, p, n) = \frac{1}{N} \left( \sum_{i=1}^N \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\} \right)

    where :math:`d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p`.

    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor
        p: the norm degree. Default: 2

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> input3 = autograd.Variable(torch.randn(100, 128))
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=False):
        super(EmbHingeLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative, target):
        return F.triplet_margin_loss(anchor, positive, negative, self.margin,
                                     self.p, self.eps, self.swap)

class EmbSquareHingeLoss(nn.Module):
    
    def __init__(self, margin=1.0, p=2, eps=1e-6 ):
        super(EmbSquareHingeLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps

    def forward(self, anchor, positive, negative, target):        
        dist_pos = F.pairwise_distance(anchor, positive, p=self.p, eps=self.eps)
        dist_neg = F.pairwise_distance(anchor, negative, p=self.p, eps=self.eps)
        triplet_loss = nn.MarginRankingLoss(margin=self.margin)(torch.pow(dist_pos, 2), torch.pow(dist_neg, 2), target)
        return triplet_loss
    
class EmbSoftHingeLoss(nn.Module):
    
    def __init__(self, margin=1.0, p=2, eps=1e-6 ):
        super(EmbSoftHingeLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps

    def forward(self, anchor, positive, negative, target):        
        dist_pos  = F.pairwise_distance(  anchor, positive, p=self.p, eps=self.eps)
        dist_neg1 = F.pairwise_distance(  anchor, negative, p=self.p, eps=self.eps)
        dist_neg2 = F.pairwise_distance(positive, negative, p=self.p, eps=self.eps)
        dist_neg_s = (torch.exp(self.margin - dist_neg1) + torch.exp(self.margin - dist_neg2))
        loss = torch.mean(torch.log(dist_neg_s) + dist_pos)
        return loss

class Accuracy(nn.Module):
    
    def __init__(self ):
        super(Accuracy, self).__init__()
        
    def forward(self, anchor, positive, negative):
        margin = 0.0
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)    
        pred = (dist_neg - dist_pos - margin).cpu().data    
        return (pred > 0).float().sum()*1.0/dist_pos.size()[0]

