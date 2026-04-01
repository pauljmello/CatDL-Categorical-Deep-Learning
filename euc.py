from dataclasses import dataclass
from typing import Callable, Union

import torch
from torch import Tensor

Obj = Union[int, tuple]  # R^n or products of objects
Data = Union[Tensor, tuple]  # tensors or tuples of data
Pullback = Callable[[Data], Data]  # cotangent map


def obj_eq(a, b):
    if type(a) is int and type(b) is int:
        return int(a) == int(b)

    if type(a) is tuple and type(b) is tuple:
        if len(a) != len(b):
            return False

        for x, y in zip(a, b):
            if not obj_eq(x, y):
                return False

        return True

    return False



def tree_map(fn, t):
    if type(t) is tuple:
        results = []
        for x in t:
            results.append(tree_map(fn, x))

        return tuple(results)

    return fn(t)


def tree_map2(fn, a, b):
    if type(a) is tuple and type(b) is tuple:
        results = []
        for x, y in zip(a, b, strict=True):
            results.append(tree_map2(fn, x, y))

        return tuple(results)

    return fn(a, b)


def tree_map3(fn, a, b, c):
    if type(a) is tuple and type(b) is tuple and type(c) is tuple:
        results = []
        for x, y, z in zip(a, b, c, strict=True):
            results.append(tree_map3(fn, x, y, z))

        return tuple(results)

    return fn(a, b, c)


def tree_flatten(t):
    out: list[Tensor] = []

    def rec(x):
        if type(x) is tuple:
            for y in x:
                rec(y)
        else:
            out.append(x)

    rec(t)
    return out


def tree_zeros_like(t):
    return tree_map(torch.zeros_like, t)


def data_zeros(obj, *, batch_shape=(), device=None, dtype=torch.float32):
    if type(obj) is int:
        return torch.zeros(*batch_shape, int(obj), device=device, dtype=dtype)

    results = []
    for o in obj:
        results.append(data_zeros(o, batch_shape=batch_shape, device=device, dtype=dtype))

    return tuple(results)


def data_add(a, b):
    if type(a) is tuple and type(b) is tuple:
        results = []
        for x, y in zip(a, b, strict=True):
            results.append(data_add(x, y))

        return tuple(results)

    return a + b


def data_scale(alpha, a):
    if type(a) is tuple:
        results = []
        for x in a:
            results.append(data_scale(alpha, x))

        return tuple(results)

    return alpha * a


def data_inner(a, b):
    if type(a) is tuple and type(b) is tuple:
        total = 0.0
        for x, y in zip(a, b, strict=True):
            total += data_inner(x, y)

        return total

    return float((a * b).sum().item())


def tensor_sum_batch(x):
    if x.ndim == 1:
        return x

    dims = tuple(range(x.ndim - 1))
    return x.sum(dim=dims)


@dataclass
class Mor:
    """
    Morphism f: dom -> cod equipped with reverse derivative.
    run(x) -> (f(x), R[f](x, -)) where R[f](x, -): cod* -> dom* is the pullback (linear in second arg).
    """
    dom: Obj
    cod: Obj
    run: Callable[[Data], tuple[Data, Pullback]]

    def __call__(self, x):
        y, _ = self.run(x)
        return y

    def __rshift__(self, g):
        return compose(self, g)

    def __matmul__(self, g):
        return prod(self, g)


def compose(f, g):
    """
    Sequential composition: f >> g = g . f.
    Chain rule: R[g.f](x, z') = R[f](x, R[g](f(x), z')).
    """
    assert obj_eq(f.cod, g.dom), f"Cannot compose: f.cod={f.cod} != g.dom={g.dom}"

    def run(x):
        y, pb_f = f.run(x)
        z, pb_g = g.run(y)

        def pullback(z_bar):
            return pb_f(pb_g(z_bar))

        return z, pullback

    return Mor(f.dom, g.cod, run)


def identity(obj):
    """
    Identity morphism: R[id](x, y') = y'.
    """

    def run(x):
        def pullback(y_bar):
            return y_bar

        return x, pullback

    return Mor(obj, obj, run)


def prod(f, g):
    """
    Product: (f x g)(a, b) = (f(a), g(b)) with tuple data.
    """

    def run(x):
        xa, xb = x
        ya, pb_f = f.run(xa)
        yb, pb_g = g.run(xb)

        def pullback(y_bar):
            ya_bar, yb_bar = y_bar
            return (pb_f(ya_bar), pb_g(yb_bar))

        return (ya, yb), pullback

    return Mor((f.dom, g.dom), (f.cod, g.cod), run)


def proj1(a, b):
    """
    First projection: pi1(x, y) = x.
    R[pi1]((x, y), x') = (x', 0).
    """

    def run(x):
        xa, xb = x

        def pullback(y_bar):
            if type(b) is int:
                z = torch.zeros_like(xb)
            else:
                z = tree_zeros_like(xb)
            return (y_bar, z)

        return xa, pullback

    return Mor((a, b), a, run)


def proj2(a, b):
    """
    Second projection: pi2(x, y) = y.
    R[pi2]((x, y), y') = (0, y').
    """

    def run(x):
        xa, xb = x

        def pullback(y_bar):
            if type(a) is int:
                z = torch.zeros_like(xa)
            else:
                z = tree_zeros_like(xa)
            return (z, y_bar)

        return xb, pullback

    return Mor((a, b), b, run)


def diagonal(obj):
    """
    Diagonal: x -> (x, x).  R[Delta](x, (a', b')) = a' + b'.
    """

    def run(x):
        def pullback(y_bar):
            a_bar, b_bar = y_bar
            return data_add(a_bar, b_bar)

        return (x, x), pullback

    return Mor(obj, (obj, obj), run)


def codiagonal(obj):
    """
    Codiagonal: (a, b) -> a + b.  R[nabla](x, y') = (y', y').
    """

    def run(x):
        a, b = x

        def pullback(y_bar):
            return (y_bar, y_bar)

        return data_add(a, b), pullback

    return Mor((obj, obj), obj, run)


def swap(a, b):
    """
    Braiding: (x, y) -> (y, x).
    """

    def run(x):
        xa, xb = x

        def pullback(y_bar):
            yb_bar, ya_bar = y_bar
            return (ya_bar, yb_bar)

        return (xb, xa), pullback

    return Mor((a, b), (b, a), run)


def terminal(obj):
    """
    Terminal morphism: A -> ().  R[!](x, ()) = 0.
    """

    def run(x):
        def pullback(y_bar):
            return tree_zeros_like(x) if isinstance(x, (Tensor, tuple)) else data_zeros(obj, device=get_device(x), dtype=get_dtype(x))

        return (), pullback

    return Mor(obj, (), run)


def get_device(x):
    if type(x) is tuple:
        if len(x) == 0:
            return None

        return get_device(x[0])

    return x.device


def get_dtype(x):
    if type(x) is tuple:
        if len(x) == 0:
            return torch.float32

        return get_dtype(x[0])

    return x.dtype


def internal_hom(a, b):
    """
    [R^m, R^n] = R^(m * n), space of linear maps A -> B.
    For leaf objects only (int dims).
    This is the object of matrices.
    """
    m, n = int(a), int(b)
    return int(m * n)


def evaluate(a, b):
    """
    eval: ([A, B], A) -> B = matrix-vector multiply.
    A linear layer IS this morphism.
    Parameters live in [A, B].
    eval((W_flat, x)) = W @ x where W = W_flat.view(n, m).
    Branch-free: W is always [n, m] (never batched).
    Input x may be single [m] or batched [B, m] - broadcasting handles both uniformly.
    Pullback sums W_bar over batch via einsum contraction.
    """
    m, n = int(a), int(b)
    hom_obj = internal_hom(a, b)

    def run(x):
        w_flat, inp = x
        W = w_flat.view(n, m)
        y = inp @ W.t()

        def pullback(y_bar):
            x_bar = y_bar @ W
            W_bar = torch.einsum('...n,...m->nm', y_bar, inp).view(n * m)
            return (W_bar, x_bar)

        return y, pullback

    return Mor((hom_obj, a), b, run)


def curry(f, c, a, b):
    """
    curry(f: (C, A) -> B): C -> [A, B].
    Analytic curry using pullbacks to extract the Jacobian df/da.
    For each output basis vector e_i, pb(e_i) gives row i of the Jacobian of f w.r.t. A.
    This is exact when f is linear in A (the CCC use case), and uses O(n) pullback calls instead of O(m) forward passes with finite differences.
    Pullback limitation: evaluates f at a = 0, so c_bars[i] (the c-component of each pullback probe) are zero for bilinear f(c, a).
    The pullback of curry w.r.t. c is therefore trivially zero for the CCC evaluation morphism.
    This is acceptable for verification (eval-curry adjunction, uncurry roundtrip) but curry must not appear in any training path where gradients through c are needed.
    """
    m, n = int(a), int(b)

    def run(x):
        device = get_device(x)
        dtype = get_dtype(x)
        zero_a = torch.zeros(m, device=device, dtype=dtype)
        basis = torch.eye(n, device=device, dtype=dtype)

        _, pb = f.run((x, zero_a))

        rows = []
        c_bars = []
        for i in range(n):
            c_bar_i, a_bar_i = pb(basis[i])
            rows.append(a_bar_i)
            c_bars.append(c_bar_i)

        W = torch.stack(rows, dim=0)
        w_flat = W.view(n * m)

        def pullback(w_bar):
            W_bar = w_bar.view(n, m)
            x_bar = data_zeros(c, device=device, dtype=dtype)
            for i in range(n):
                scale = float(W_bar[i].sum())
                if abs(scale) > 1e-12:
                    x_bar = data_add(x_bar, data_scale(scale, c_bars[i]))

            return x_bar

        return w_flat, pullback

    return Mor(c, internal_hom(a, b), run)


def uncurry(f, c, a, b):
    """
    uncurry(f: C -> [A, B]): (C, A) -> B = f >> eval.
    """
    ev = evaluate(a, b)

    def run(x):
        xc, xa = x
        w_flat, pb_f = f.run(xc)
        y, pb_ev = ev.run((w_flat, xa))

        def pullback(y_bar):
            w_bar, xa_bar = pb_ev(y_bar)
            xc_bar = pb_f(w_bar)
            return (xc_bar, xa_bar)

        return y, pullback

    return Mor((c, a), b, run)


def batch_shape(x):
    """
    Extract leading batch dimensions from data.
    """
    if type(x) is Tensor and x.dim() > 1:
        return x.shape[:-1]

    return ()


def zero_mor(a, b):
    """
    Zero morphism: always returns 0.
    Pullback is zero.
    """

    def run(x):
        device, dtype = get_device(x), get_dtype(x)
        y = data_zeros(b, batch_shape=batch_shape(x), device=device, dtype=dtype)

        def pullback(y_bar):
            return data_zeros(a, batch_shape=batch_shape(y_bar), device=device, dtype=dtype)

        return y, pullback

    return Mor(a, b, run)


def add_mor(f, g):
    """
    (f + g)(x) = f(x) + g(x).
    Pullback sums.
    """

    def run(x):
        yf, pb_f = f.run(x)
        yg, pb_g = g.run(x)

        def pullback(y_bar):
            return data_add(pb_f(y_bar), pb_g(y_bar))

        return data_add(yf, yg), pullback

    return Mor(f.dom, f.cod, run)


def scale_mor(alpha, f):
    """
    (alpha * f)(x) = alpha * f(x).
    Pullback scales.
    """

    def run(x):
        y, pb_f = f.run(x)

        def pullback(y_bar):
            return data_scale(alpha, pb_f(y_bar))

        return data_scale(alpha, y), pullback

    return Mor(f.dom, f.cod, run)


def forward_deriv(f, x, v, eps=1e-5):
    """
    D[f](x)(v) via central finite differences: (f(x + eps * v) - f(x - eps * v)) / (2 * eps).
    For verification of JVP-VJP duality.
    """
    x_plus = data_add(x, data_scale(eps, v))
    x_minus = data_add(x, data_scale(-eps, v))
    y_plus = f(x_plus)
    y_minus = f(x_minus)
    return data_scale(1.0 / (2.0 * eps), data_add(y_plus, data_scale(-1.0, y_minus)))


def residual(f):
    """
    Residual: Delta ; (id x f) ; nabla.
    res(f)(x) = x + f(x).
    """
    return compose(compose(diagonal(f.dom), prod(identity(f.dom), f)), codiagonal(f.cod))


def fanout(f, g):
    """
    Fan-out: <f, g> = Delta ; (f x g).
    <f, g>(x) = (f(x), g(x)).
    """
    return compose(diagonal(f.dom), prod(f, g))


def check_pullback_linearity(f, x, v1, v2, alpha=0.7):
    """
    Verify R[f](x, alpha * v1 + v2) = alpha * R[f](x, v1) + R[f](x, v2).
    """
    _, pb = f.run(x)
    lhs = pb(data_add(data_scale(alpha, v1), v2))
    rhs = data_add(data_scale(alpha, pb(v1)), pb(v2))
    lhs_flat = tree_flatten(lhs)
    rhs_flat = tree_flatten(rhs)
    for a, b in zip(lhs_flat, rhs_flat):
        if not torch.allclose(a, b, rtol=1e-4, atol=1e-4):
            return False

    return True


def check_chain_rule(f, g, x, z_bar):
    """
    Verify R[g.f](x, z') = R[f](x, R[g](f(x), z')).
    """
    fg = compose(f, g)
    _, pb_fg = fg.run(x)
    result_composed = pb_fg(z_bar)

    y, pb_f = f.run(x)
    _, pb_g = g.run(y)
    result_chain = pb_f(pb_g(z_bar))

    lhs_flat = tree_flatten(result_composed)
    rhs_flat = tree_flatten(result_chain)
    for a, b in zip(lhs_flat, rhs_flat):
        if not torch.allclose(a, b, rtol=1e-4, atol=1e-4):
            return False

    return True


def check_jvp_vjp_duality(f, x, v, w, eps=1e-4):
    """
    Verify <D[f](x)(v), w> = <v, R[f](x)(w)> (JVP-VJP duality).
    Uses finite-difference JVP and analytic VJP.
    """
    jvp = forward_deriv(f, x, v, eps=eps)
    _, pb = f.run(x)
    vjp_result = pb(w)

    lhs = data_inner(jvp, w)
    rhs = data_inner(v, vjp_result)

    return abs(lhs - rhs) < 0.05 * (abs(lhs) + abs(rhs) + 1e-8)
