from dataclasses import dataclass
from typing import Callable

import torch

from euc import Obj, Data, Mor, swap, tensor_sum_batch, internal_hom, evaluate


@dataclass
class Para:
    """
    Parameterized morphism in Para(Euc).
    p: parameter object (honest tuple Obj).
    dom: input object.
    cod: output object.
    run: (params, x) -> (y, pullback) where pullback(y') -> (param_grads, x').
    init_fn: () -> initial params.
    """
    p: Obj
    dom: Obj
    cod: Obj
    run: Callable[[Data, Data], tuple[Data, Callable[[Data], tuple[Data, Data]]]]
    init_fn: Callable[..., Data]

    def __call__(self, params, x):
        y, _ = self.run(params, x)
        return y

    def init(self, *, device=None, dtype=torch.float32, generator=None):
        return self.init_fn(device=device, dtype=dtype, generator=generator)

    def __rshift__(self, g):
        return para_compose(self, g)

    def __matmul__(self, g):
        return para_prod(self, g)


def para_compose(f, g):
    """
    Composition: params = (pf, pg) as honest tuple.
    """
    from euc import obj_eq
    assert obj_eq(f.cod, g.dom), f"Cannot compose: f.cod={f.cod} != g.dom={g.dom}"

    def run(params, x):
        pf, pg = params
        y, pb_f = f.run(pf, x)
        z, pb_g = g.run(pg, y)

        def pullback(z_bar):
            pg_bar, y_bar = pb_g(z_bar)
            pf_bar, x_bar = pb_f(y_bar)
            return (pf_bar, pg_bar), x_bar

        return z, pullback

    def init(*, device=None, dtype=torch.float32, generator=None):
        return (
            f.init(device=device, dtype=dtype, generator=generator),
            g.init(device=device, dtype=dtype, generator=generator),
        )

    return Para((f.p, g.p), f.dom, g.cod, run, init)


def para_identity(obj):
    """
    Identity in Para: empty params ().
    """

    def run(params, x):
        def pullback(y_bar):
            return (), y_bar

        return x, pullback

    def init(*, device=None, dtype=torch.float32, generator=None):
        return ()

    return Para((), obj, obj, run, init)


def para_prod(f, g):
    """
    Parallel product: params = (pl, pr), input/output = tuples.
    """

    def run(params, x):
        pl, pr = params
        xl, xr = x
        yl, pb_l = f.run(pl, xl)
        yr, pb_r = g.run(pr, xr)

        def pullback(y_bar):
            yl_bar, yr_bar = y_bar
            gl, xl_bar = pb_l(yl_bar)
            gr, xr_bar = pb_r(yr_bar)
            return (gl, gr), (xl_bar, xr_bar)

        return (yl, yr), pullback

    def init(*, device=None, dtype=torch.float32, generator=None):
        return (
            f.init(device=device, dtype=dtype, generator=generator),
            g.init(device=device, dtype=dtype, generator=generator),
        )

    return Para((f.p, g.p), (f.dom, g.dom), (f.cod, g.cod), run, init)



def para_swap(a, b):
    """
    Structural lift of swap to Para.
    """
    return lift(swap(a, b))


def to_mor(p, params):
    """
    Freeze params, get base category morphism.
    Forgetful functor Para -> Euc.
    """

    def run(x):
        y, pb = p.run(params, x)

        def pullback(y_bar):
            _, x_bar = pb(y_bar)
            return x_bar

        return y, pullback

    return Mor(p.dom, p.cod, run)


def lift(f):
    """
    Lift parameter-free morphism into Para.
    """

    def run(params, x):
        y, pb_f = f.run(x)

        def pullback(y_bar):
            x_bar = pb_f(y_bar)
            return (), x_bar

        return y, pullback

    def init(*, device=None, dtype=torch.float32, generator=None):
        return ()

    return Para((), f.dom, f.cod, run, init)


def para_from_eval(a, b, bias=True):
    """
    THE key function: Affine layer = eval from CCC.
    Parameters live in [A, B] = R^(m * n) (weight matrix).
    If bias = True, params = (W_flat, b) where W in [A, B] and b in B.
    The forward pass delegates to the CCC evaluation morphism eval: ([A, B], A) -> B from euc.py.
    A linear layer IS this morphism - the CCC origin is explicit in code, not just docs.
    """
    m, n = int(a), int(b)
    ev = evaluate(a, b)
    hom_obj = internal_hom(a, b)

    if bias:
        p_obj: Obj = (hom_obj, b)
    else:
        p_obj = (hom_obj,)

    def run(params, x):
        if bias:
            w_flat, b_vec = params
        else:
            (w_flat,) = params
            b_vec = None

        y, pb_ev = ev.run((w_flat, x))

        if b_vec is not None:
            y = y + b_vec

        def pullback(y_bar):
            w_bar, x_bar = pb_ev(y_bar)

            if b_vec is not None:
                b_bar = tensor_sum_batch(y_bar)
                return (w_bar, b_bar), x_bar

            return (w_bar,), x_bar

        return y, pullback

    def init(*, device=None, dtype=torch.float32, generator=None):
        std = 1.0 / (m ** 0.5)
        w = torch.randn(n * m, generator=generator, device=device, dtype=dtype) * std

        if bias:
            b_vec = torch.zeros(n, device=device, dtype=dtype)
            return (w, b_vec)

        return (w,)

    return Para(p_obj, a, b, run, init)


def sequential(*layers):
    """
    Compose chain: l1 >> l2 >> ... >> ln.
    """
    result = layers[0]
    for layer in layers[1:]:
        result = para_compose(result, layer)

    return result
