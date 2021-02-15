"""
.. module:: SecmlAutograd
    :synopsis: Wraps a secML CModule or chain of CModules inside
    a PyTorch autograd layer.

.. moduleauthor:: Luca Demetrio <luca.demetrio@dibris.unige.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
import torch
from secml.array import CArray
from torch import nn


class SecmlAutogradFunction(torch.autograd.Function):
    """
    This class wraps a generic secML classifier inside a PyTorch autograd function.
    When the function's backward is called, the secML module calls the internal
    backward of the CModule, and links it to the external graph.
    Reference here: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, clf):
        ctx.clf = clf
        ctx.save_for_backward(input)
        out = as_tensor(clf.decision_function(as_carray(input.view(input.shape[0], -1))))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        clf = ctx.clf
        input, = ctx.saved_tensors
        # https://github.com/pytorch/pytorch/issues/1776#issuecomment-372150869
        with torch.enable_grad():
            grad_input = clf.gradient(x=as_carray(input),
                                      w=as_carray(grad_output))

        grad_input = as_tensor(grad_input, True)
        if hasattr(clf, 'input_shape'):
            input_shape = clf.input_shape
        else:
            input_shape = [clf.n_features]
        if len(grad_input.shape) > 1:
            grad_input = grad_input.view((grad_input.shape[0], *input_shape))
        else:
            grad_input = grad_input.view((1, *input_shape))
        return grad_input, None


def as_tensor(x, requires_grad=False):
    if isinstance(x, CArray):
        x = x.tondarray()
    x = torch.from_numpy(x)
    x.requires_grad = requires_grad
    return x


def as_carray(x):
    # FIXME this is nasty!
    if len(x.shape) > 2:
        x = x.flatten(start_dim=1)
    return CArray(x.cpu().detach().numpy())


class SecmlLayer(nn.Module):
    def __init__(self, model):
        super(SecmlLayer, self).__init__()
        self._clf = model
        self.secml_autograd = SecmlAutogradFunction.apply
        self.eval()

    def forward(self, x):
        x = self.secml_autograd(x, self._clf)
        return x

    def extra_repr(self) -> str:
        return "Wrapper of SecML model {}".format(self._clf)
