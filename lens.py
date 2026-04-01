from dataclasses import dataclass
from typing import Callable


@dataclass
class Lens:
    """
    Lens (optic) with get and put.
    get: S -> (A, Residual).
    put: (Residual, A') -> S'.
    """

    get: Callable
    put: Callable


def lens_id():
    """
    Identity lens.
    """

    def get(s):
        return (s, None)

    def put(res, a_prime):
        return a_prime

    return Lens(get=get, put=put)


def lens_compose(l1, l2):
    """
    Compose two lenses: l1 ; l2.
    """

    def get(s):
        a, res1 = l1.get(s)
        b, res2 = l2.get(a)
        return b, (res1, res2)

    def put(res, b_prime):
        res1, res2 = res
        a_prime = l2.put(res2, b_prime)
        return l1.put(res1, a_prime)

    return Lens(get=get, put=put)


def lens_prod(l1, l2):
    """
    Product of two lenses.
    """

    def get(s):
        s1, s2 = s
        a1, res1 = l1.get(s1)
        a2, res2 = l2.get(s2)
        return (a1, a2), (res1, res2)

    def put(res, a_prime):
        res1, res2 = res
        a1_prime, a2_prime = a_prime
        return (l1.put(res1, a1_prime), l2.put(res2, a2_prime))

    return Lens(get=get, put=put)


def model_lens(model, x):
    """
    Forward pass as lens get, backward as put.
    get: params -> (logits, residual=pullback).
    put: (pullback, d_logits) -> param_grads.
    """

    def get(p):
        logits, pullback = model.run(p, x)
        return logits, pullback

    def put(pullback, d_logits):
        param_grads, _ = pullback(d_logits)
        return param_grads

    return Lens(get=get, put=put)


def loss_lens(loss_fn, labels):
    """
    Loss computation as lens.
    get: logits -> (loss, residual=(d_logits,)).
    put: (residual, _) -> d_logits.
    """

    def get(logits):
        loss, d_logits = loss_fn.loss_and_grad(logits, labels)
        return loss, d_logits

    def put(d_logits, loss_bar):
        if loss_bar is None:
            return d_logits

        return loss_bar * d_logits

    return Lens(get=get, put=put)


def training_step(model, loss_fn, opt, x, y, params, state, lr):
    """
    One training step as composed optic: model_lens >> loss_lens.
    The chain rule is a consequence of composing optics via lens_compose, not manual pullback wiring.
    The optimizer stays outside the optic (it's stateful, not a lens).
    Returns: (new_params, loss, new_state).
    """

    ml = model_lens(model, x)
    ll = loss_lens(loss_fn, y)
    step = lens_compose(ml, ll)

    loss, residual = step.get(params)
    param_grads = step.put(residual, None)

    new_params, new_state = opt.step(params, state, param_grads, lr)
    return new_params, loss, new_state
