import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        groups,
        reset_gate=True,
        min_reset=0.0,
        update_gate=True,
        min_update=0.0,
        out_bias=True,
        out_act=None,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.padding = padding
        self.input_size = input_size
        self.hidden_size = hidden_size
        if reset_gate:
            self.min_reset = min_reset
            self.reset_gate = nn.Conv2d(
                input_size + hidden_size,
                hidden_size,
                kernel_size,
                groups=groups,
                padding=padding,
            )
        else:
            self.reset_gate = None

        if update_gate:
            self.min_update = min_update
            self.update_gate = nn.Conv2d(
                input_size + hidden_size,
                hidden_size,
                kernel_size,
                groups=groups,
                padding=padding,
            )
        else:
            self.update_gate = None

        # self.out_gate = nn.Conv2d(
        #     input_size + hidden_size,
        #     hidden_size,
        #     kernel_size,
        #     groups=groups,
        #     padding=padding,
        #     bias=out_bias,
        # )

        W = torch.ones(
            hidden_size, hidden_size, kernel_size, kernel_size
        )
        self.out_weights = nn.Parameter(
            0.75 * W / input_size / kernel_size ** 2
        )

        if self.reset_gate:
            init.orthogonal_(self.reset_gate.weight)
            init.constant_(self.reset_gate.bias, 0.0)
        if self.update_gate:
            init.orthogonal_(self.update_gate.weight)
            init.constant_(self.update_gate.bias, 0.0)
        # init.orthogonal_(self.out_gate.weight)
        # eye = torch.eye(kernel_size, kernel_size).unsqueeze(0).unsqueeze(0)
        # init.constant_(self.out_gate.weight, eye)
        if out_bias:
            init.constant_(self.out_gate.bias, 0.0)

        if out_act is None:
            self.out_act = None
        elif out_act == "tanh":
            self.out_act = torch.tanh
        elif out_act == "leaky_relu":
            self.out_act = torch.leaky_relu
        else:
            raise NotImplementedError(out_act)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        # must intercalate instead of concatenate !!!!
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        # interleaved_inputs = interleave_filters(input_, prev_state)

        if self.reset_gate:
            reset = torch.sigmoid(self.reset_gate(stacked_inputs))
            reset = self.min_reset + (1.0 - self.min_reset) * reset
        else:
            reset = torch.ones_like(prev_state)

        # convolution wiht positive weights
        # out_inputs = torch.cat([input_, prev_state * reset], dim=1)
        out_inputs = prev_state * reset
        # W = F.softplus(self.log_out_weights)
        out_inputs = out_inputs + input_
        out_inputs = F.conv2d(
            out_inputs, self.out_weights, padding=self.padding
        )

        if self.out_act is not None:
            out_inputs = self.out_act(out_inputs)

        if self.update_gate:
            update = torch.sigmoid(self.update_gate(stacked_inputs))
            update = self.min_update + (1.0 - self.min_update) * update
        else:
            update = torch.ones_like(prev_state)

        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


def interleave_filters(a, b):
    filt_a = a.shape[1]
    filt_b = b.shape[1]
    assert filt_b % filt_a == 0
    filt_ratio = filt_b // filt_a
    batch_size = a.shape[0]
    spatial_size = a.shape[2:]
    newdim = (batch_size, filt_a + filt_b, *spatial_size)
    out = torch.empty(newdim)
    for f in range(filt_a):
        out[:, f * filt_ratio] = a[:, f]
        b_dst = b[:, f : (f + filt_ratio)]
        out[:, (f * filt_ratio + 1) : ((f + 1) * filt_ratio)] = b_dst
    return out


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        """
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        """

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            assert (
                len(hidden_sizes) == n_layers
            ), "`hidden_sizes` must have the same length as n_layers"
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert (
                len(kernel_sizes) == n_layers
            ), "`kernel_sizes` must have the same length as n_layers"
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvGRUCell(
                input_dim, self.hidden_sizes[i], self.kernel_sizes[i]
            )
            name = "ConvGRUCell_" + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        """
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        """
        if not hidden:
            hidden = [None] * self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden
