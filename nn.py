import torch
import torch.nn.functional as F

from euc import Obj, Mor, identity, diagonal, codiagonal
from para import Para, para_prod, para_from_eval, lift, sequential


def ensure_batch(x):
    """
    Add batch dim if missing.
    Returns (x_batched, was_single).
    """
    if x.dim() == 1:
        return x.unsqueeze(0), True

    return x, False


def maybe_unbatch(y, single):
    """
    Remove batch dim if input was single.
    """
    if single:
        return y.squeeze(0)

    return y


def Affine(in_dim, out_dim, bias=True):
    """
    Affine map x -> Wx + b.
    This IS para_from_eval from the CCC.
    """
    return para_from_eval(int(in_dim), int(out_dim), bias=bias)


def ReLU(n):
    """
    ReLU: max(0, x).
    R[relu](x, y') = y' * 1_{x > 0}.
    """
    dim = int(n)

    def run_mor(x):
        y = torch.relu(x)
        mask = (x > 0).to(x.dtype)

        def pullback(y_bar):
            return y_bar * mask

        return y, pullback

    return lift(Mor(dim, dim, run_mor))


def Tanh(n):
    """
    Tanh: tanh(x).
    R[tanh](x, y') = y' * (1 - tanh(x)^2).
    """
    dim = int(n)

    def run_mor(x):
        y = torch.tanh(x)

        def pullback(y_bar):
            return y_bar * (1.0 - y * y)

        return y, pullback

    return lift(Mor(dim, dim, run_mor))


def Softplus(n, beta=10.0):
    """
    Softplus: softplus(beta * x) / beta.
    Smooth ReLU surrogate.
    """
    dim = int(n)
    beta = float(beta)

    def run_mor(x):
        y = F.softplus(x * beta) / beta
        s = torch.sigmoid(x * beta)

        def pullback(y_bar):
            return y_bar * s

        return y, pullback

    return lift(Mor(dim, dim, run_mor))



def Conv2d(in_ch, out_ch, H, W, kernel=3, padding=1, bias=True):
    """
    2D Convolution: translation-equivariant morphism.
    Input/output are flattened: R^(in_ch * H * W) -> R^(out_ch * H * W).
    Parameters: kernel in R^(out_ch * in_ch * k * k), bias in R^out_ch.
    """
    in_dim = in_ch * H * W
    out_dim = out_ch * H * W
    kernel_numel = out_ch * in_ch * kernel * kernel

    if bias:
        p_obj: Obj = (kernel_numel, out_ch)
    else:
        p_obj = (kernel_numel,)

    def run(params, x):
        if bias:
            k_flat, b_vec = params
        else:
            (k_flat,) = params
            b_vec = None

        K = k_flat.view(out_ch, in_ch, kernel, kernel)

        x, single = ensure_batch(x)
        batch = x.shape[0]
        x_img = x.view(batch, in_ch, H, W)

        y_img = F.conv2d(x_img, K, bias=b_vec, padding=padding)
        y = maybe_unbatch(y_img.view(batch, -1), single)

        def pullback(y_bar):
            y_bar, _ = ensure_batch(y_bar)
            yb = y_bar.view(batch, out_ch, H, W)

            x_bar_img = F.conv_transpose2d(yb, K, padding=padding)
            x_bar = maybe_unbatch(x_bar_img.view(batch, -1), single)

            x_unfold = F.unfold(x_img, kernel_size=kernel, padding=padding)
            yb_flat = yb.view(batch, out_ch, -1)
            K_bar = torch.einsum('bos,bps->op', yb_flat, x_unfold)
            K_bar = K_bar.view(out_ch, in_ch, kernel, kernel)
            k_bar = K_bar.flatten()

            if bias:
                b_bar = yb.sum(dim=(0, 2, 3))
                return (k_bar, b_bar), x_bar

            return (k_bar,), x_bar

        return y, pullback

    def init(*, device=None, dtype=torch.float32, generator=None):
        fan_in = in_ch * kernel * kernel
        std = 1.0 / (fan_in ** 0.5)
        k = torch.randn(kernel_numel, generator=generator, device=device, dtype=dtype) * std

        if bias:
            b_vec = torch.zeros(out_ch, device=device, dtype=dtype)
            return (k, b_vec)

        return (k,)

    return Para(p_obj, in_dim, out_dim, run, init)


def MaxPool2d(in_ch, H, W, kernel=2):
    """
    Max pooling: parameter-free spatial downsampling.
    """
    in_dim = in_ch * H * W
    out_H, out_W = H // kernel, W // kernel
    out_dim = in_ch * out_H * out_W

    def run_mor(x):
        x, single = ensure_batch(x)
        batch = x.shape[0]
        x_img = x.view(batch, in_ch, H, W)
        y_img, indices = F.max_pool2d(x_img, kernel_size=kernel, return_indices=True)
        y = maybe_unbatch(y_img.view(batch, -1), single)

        def pullback(y_bar):
            y_bar, _ = ensure_batch(y_bar)
            yb = y_bar.view(batch, in_ch, out_H, out_W)
            x_bar_img = F.max_unpool2d(yb, indices, kernel_size=kernel, output_size=(batch, in_ch, H, W))
            x_bar = maybe_unbatch(x_bar_img.view(batch, -1), single)
            return x_bar

        return y, pullback

    return lift(Mor(in_dim, out_dim, run_mor))


def GlobalAvgPool(in_ch, H, W):
    """
    Global average pooling: spatial -> channel.
    """
    in_dim = in_ch * H * W
    out_dim = in_ch
    spatial = H * W

    def run_mor(x):
        x, single = ensure_batch(x)
        batch = x.shape[0]
        x_img = x.view(batch, in_ch, H, W)
        y = maybe_unbatch(x_img.mean(dim=(2, 3)), single)

        def pullback(y_bar):
            y_bar, _ = ensure_batch(y_bar)
            yb = y_bar.view(batch, in_ch, 1, 1)
            x_bar = yb.expand(batch, in_ch, H, W).reshape(batch, -1) / spatial
            x_bar = maybe_unbatch(x_bar, single)
            return x_bar

        return y, pullback

    return lift(Mor(in_dim, out_dim, run_mor))


def mlp(in_dim, hidden, out_dim, *, act="tanh", bias=True):
    """
    MLP: chain of Affine >> Activation.
    """
    hs = list(hidden)
    act_map = {"tanh": Tanh, "relu": ReLU, "softplus": Softplus}
    act_fn = act_map[act.lower()]

    layers: list[Para] = []
    prev = int(in_dim)
    for h in hs:
        layers.append(Affine(prev, h, bias=bias))
        layers.append(act_fn(h))
        prev = h

    layers.append(Affine(prev, int(out_dim), bias=bias))

    return sequential(*layers)


def cnn(in_ch, H, W, channels, num_classes, *, act="relu"):
    """
    CNN: Conv >> Act >> Pool chain, then GlobalAvgPool >> Linear.
    """
    chs = list(channels)
    act_map = {"relu": ReLU, "tanh": Tanh}
    act_fn = act_map[act.lower()]

    layers: list[Para] = []
    curr_ch, curr_H, curr_W = in_ch, H, W

    for ch in chs:
        layers.append(Conv2d(curr_ch, ch, curr_H, curr_W, kernel=3, padding=1))
        layers.append(act_fn(ch * curr_H * curr_W))
        layers.append(MaxPool2d(ch, curr_H, curr_W, kernel=2))
        curr_ch = ch
        curr_H, curr_W = curr_H // 2, curr_W // 2

    layers.append(GlobalAvgPool(curr_ch, curr_H, curr_W))

    layers.append(Affine(curr_ch, num_classes))

    return sequential(*layers)


def residual_block(f):
    """
    Residual: Delta ; (id x f) ; nabla in Para.
    """
    x_obj = f.dom
    dup = lift(diagonal(x_obj))
    add = lift(codiagonal(x_obj))
    id_x = lift(identity(x_obj))
    return sequential(dup, para_prod(id_x, f), add)


def residual_mlp(dim, depth, *, act="tanh", bias=True):
    """
    Stacked residual blocks: R^dim -> R^dim.
    """
    d = int(dim)
    act_map = {"tanh": Tanh, "relu": ReLU, "softplus": Softplus}
    act_fn = act_map[act.lower()]

    def block():
        return sequential(Affine(d, d, bias=bias), act_fn(d), Affine(d, d, bias=bias))

    layers = []
    for _ in range(depth):
        layers.append(residual_block(block()))

    return sequential(*layers)
