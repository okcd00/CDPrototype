import torch
from torch.nn import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    MSELoss,
    NLLLoss
)
from focal_loss import FocalLoss


_REDUCTION = 'mean'
LOSS_FN_LIB = {
    # -w_i(y_i \cdot \log x_i + (1-y_i) \ log (1-x_i))
    'bce': BCELoss(  # (sigmoid(logits).squeeze(-1), target.double())
        reduction=_REDUCTION),
    'bcel': BCEWithLogitsLoss(  # (logits.squeeze(-1), target.float())
        reduction=_REDUCTION),
    'ce': CrossEntropyLoss(  # (logits, target.long())
        ignore_index=0,  # 0 for [PAD]
        reduction=_REDUCTION),  
    # - \alpha_t \cdot (1-p_t)^{\gamma} \log (p_t)
    'focal': FocalLoss(  # (logits.squeeze(-1), target)
        num_labels=None,  # alpha=-1, 
        activation_type='sigmoid',
        reduction=_REDUCTION),
    'mse': MSELoss(  # (sigmoid(logits).squeeze(-1), target)
        reduction=_REDUCTION),
    'nll': NLLLoss(  # (log(softmax(logits)), target.long())
        ignore_index=-1, 
        reduction=_REDUCTION),
}


def unit_test():
    eps = 1e-8
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(-1)

    import numpy as np
    # logits (batch_size, 1) 
    logits_binary = torch.from_numpy(np.array([[1e3], [-2e3], [3e3]]))
    # logits (batch_size, num_classes)
    logits_nclass = torch.from_numpy(np.array([[-1e3, 1e3], [2e3, 0.], [0., 2e3]]))
    # labels (batch_size)
    labels = torch.from_numpy(np.array([0, 1, 1]))
    # labels = torch.from_numpy(np.array([1, 0, 1]))

    for loss_name, loss_fn in LOSS_FN_LIB.items():
        if loss_name in ['bce']:
            # (batch_size), (batch_size)
            _logits = sigmoid(logits_binary).squeeze(-1)
            _labels = labels.double()
        elif loss_name in ['bcel']:
            # (batch_size), (batch_size)
            _logits = logits_binary.squeeze(-1)
            _labels = labels.float()
        elif loss_name in ['ce']:
            # (batch_size, num_classes), (batch_size)
            _logits = logits_nclass
            _labels = labels.long()
        elif loss_name in ['focal']:
            # (batch_size, num_classes), (batch_size)
            _logits = logits_binary.squeeze(-1)
            _labels = labels  # bool() long() float() double()
        elif loss_name in ['mse']:
            _logits = sigmoid(logits_binary).squeeze(-1)
            _labels = labels  # bool() long() float() double()
        elif loss_name in ['nll']:
            _logits = torch.log(softmax(logits_nclass) + eps)
            _labels = labels.long()
        print(loss_name, loss_fn(_logits, _labels).item())


if __name__ == "__main__":
    unit_test()
