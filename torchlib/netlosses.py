import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def to_gpu( x, cuda ):
    return x.cuda() if cuda else x

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels.long()]     # [N,D]



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


## Baseline clasification

class TopkAccuracy(nn.Module):
    
    def __init__(self, topk=(1,)):
        super(TopkAccuracy, self).__init__()
        self.topk = topk

    def forward(self, output, target):
        """Computes the precision@k for the specified values of k"""
        
        maxk = max(self.topk)
        batch_size = target.size(0)

        pred = output.topk(maxk, 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append( correct_k.mul_(100.0 / batch_size) )

        return res



class ConfusionMeter( object ):
    """Maintains a confusion matrix for a given calssification problem.
    https://github.com/pytorch/tnt/tree/master/torchnet/meter

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """

        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

