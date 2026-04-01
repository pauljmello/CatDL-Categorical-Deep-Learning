import copy

import torch

from euc import (Mor, compose, identity, prod, diagonal, codiagonal, swap, terminal, internal_hom, evaluate, curry, uncurry, zero_mor, add_mor, scale_mor, proj1,
                 proj2, fanout, residual, data_add, data_scale, check_pullback_linearity, check_chain_rule, check_jvp_vjp_duality, tree_flatten, tree_map)
from lens import Lens, lens_id, lens_compose, lens_prod, training_step
from nn import Affine, mlp, residual_mlp, Conv2d
from para import para_compose, para_identity, para_prod, para_swap, to_mor, lift
from train import SoftmaxCrossEntropy, AdamW

EQ = dict(rtol=1e-7, atol=1e-7)
LIN = dict(rtol=1e-5, atol=1e-6)
AD = dict(rtol=2e-4, atol=2e-5)


def rng(seed):
    return torch.Generator().manual_seed(seed)


def linear_mor(in_d, out_d, *, gen):
    W = torch.randn(out_d, in_d, generator=gen)

    def run(x):
        def pb(ybar):
            return ybar @ W

        return x @ W.t(), pb

    return Mor(in_d, out_d, run), W


def check(name, got, exp, tol=EQ):
    got_flat, exp_flat = tree_flatten(got), tree_flatten(exp)
    assert len(got_flat) == len(exp_flat), f"{name}: tree length mismatch {len(got_flat)} vs {len(exp_flat)}"
    err = max((float((g - e).abs().max()) for g, e in zip(got_flat, exp_flat)), default=0.0)
    ok = all(torch.allclose(g, e, **tol) for g, e in zip(got_flat, exp_flat))
    if ok:
        print(f"  PASS: {name} (max error = {err:.3e})")
    else:
        print(f"  FAIL: {name} (max error = {err:.3e})")

    assert ok, f"{name}: max error = {err:.3e}"


def autograd_vjp(model, params, x, ybar):
    p_req = tree_map(lambda t: t.detach().clone().requires_grad_(True), params)
    x_req = x.detach().clone().requires_grad_(True)
    out = model(p_req, x_req)
    (out * ybar).sum().backward()
    return out.detach(), tree_map(lambda t: t.grad.detach(), p_req), x_req.grad.detach()


def test_crdc():
    print("\n[TEST] CRDC axioms")
    gen = rng(2)

    batch, in_d, out_d = 10, 6, 4
    f = Affine(in_d, out_d)
    params = f.init(generator=rng(20))
    x = torch.randn(batch, in_d, generator=gen)
    ybar = torch.randn(batch, out_d, generator=gen)

    out, pb = f.run(params, x)
    out_lhs, pb_lhs = para_compose(para_identity(in_d), f).run(((), params), x)
    out_rhs, pb_rhs = para_compose(f, para_identity(out_d)).run((params, ()), x)
    check("id>>f fwd", out_lhs, out)
    check("f>>id fwd", out_rhs, out)

    pgrad, xgrad = pb(ybar)
    (_, pgrad_lhs), xgrad_lhs = pb_lhs(ybar)
    (pgrad_rhs, _), xgrad_rhs = pb_rhs(ybar)
    check("id>>f pbar", pgrad_lhs, pgrad)
    check("f>>id pbar", pgrad_rhs, pgrad)
    check("id>>f xbar", xgrad_lhs, xgrad)
    check("f>>id xbar", xgrad_rhs, xgrad)

    dim_a, dim_b, dim_c, dim_d = 5, 6, 4, 3
    f, g, h = Affine(dim_a, dim_b), mlp(dim_b, [8], dim_c, act="tanh"), Affine(dim_c, dim_d)
    params_f, params_g, params_h = f.init(generator=rng(31)), g.init(generator=rng(32)), h.init(generator=rng(33))
    x = torch.randn(7, dim_a, generator=rng(3))
    zbar = torch.randn(7, dim_d, generator=rng(3))

    lhs = para_compose(para_compose(f, g), h)
    rhs = para_compose(f, para_compose(g, h))
    out_lhs, pb_lhs = lhs.run(((params_f, params_g), params_h), x)
    out_rhs, pb_rhs = rhs.run((params_f, (params_g, params_h)), x)
    check("assoc fwd", out_lhs, out_rhs)
    pgrad_lhs, xgrad_lhs = pb_lhs(zbar)
    pgrad_rhs, xgrad_rhs = pb_rhs(zbar)
    (grad_f_lhs, grad_g_lhs), grad_h_lhs = pgrad_lhs
    grad_f_rhs, (grad_g_rhs, grad_h_rhs) = pgrad_rhs
    check("assoc pbar.f", grad_f_lhs, grad_f_rhs)
    check("assoc pbar.g", grad_g_lhs, grad_g_rhs)
    check("assoc pbar.h", grad_h_lhs, grad_h_rhs)
    check("assoc xbar", xgrad_lhs, xgrad_rhs)

    f, g = Affine(3, 5), Affine(4, 6)
    params_f, params_g = f.init(generator=rng(41)), g.init(generator=rng(42))
    x_a = torch.randn(8, 3, generator=rng(4))
    x_b = torch.randn(8, 4, generator=rng(4))

    out_prod, _ = para_prod(f, g).run((params_f, params_g), (x_a, x_b))
    check("bifunctor left", out_prod[0], f(params_f, x_a))
    check("bifunctor right", out_prod[1], g(params_g, x_b))

    f_a, g_a = Affine(3, 5), Affine(5, 2)
    f_b, g_b = Affine(4, 6), Affine(6, 7)
    params_fa, params_ga = f_a.init(generator=rng(51)), g_a.init(generator=rng(52))
    params_fb, params_gb = f_b.init(generator=rng(53)), g_b.init(generator=rng(54))
    x_a = torch.randn(9, 3, generator=rng(5))
    x_b = torch.randn(9, 4, generator=rng(5))
    ybar_a = torch.randn(9, 2, generator=rng(5))
    ybar_b = torch.randn(9, 7, generator=rng(5))

    lhs = para_prod(para_compose(f_a, g_a), para_compose(f_b, g_b))
    rhs = para_compose(para_prod(f_a, f_b), para_prod(g_a, g_b))
    out_lhs, pb_lhs = lhs.run(((params_fa, params_ga), (params_fb, params_gb)), (x_a, x_b))
    out_rhs, pb_rhs = rhs.run(((params_fa, params_fb), (params_ga, params_gb)), (x_a, x_b))
    check("interchange fwd", out_lhs, out_rhs)

    pgrad_lhs, xgrad_lhs = pb_lhs((ybar_a, ybar_b))
    pgrad_rhs, xgrad_rhs = pb_rhs((ybar_a, ybar_b))
    (grad_fa_lhs, grad_ga_lhs), (grad_fb_lhs, grad_gb_lhs) = pgrad_lhs
    (grad_fa_rhs, grad_fb_rhs), (grad_ga_rhs, grad_gb_rhs) = pgrad_rhs
    check("interchange pbar.fl", grad_fa_lhs, grad_fa_rhs)
    check("interchange pbar.gl", grad_ga_lhs, grad_ga_rhs)
    check("interchange pbar.fr", grad_fb_lhs, grad_fb_rhs)
    check("interchange pbar.gr", grad_gb_lhs, grad_gb_rhs)
    check("interchange xbar", xgrad_lhs, xgrad_rhs)


def test_cartesian():
    print("\n[TEST] Cartesian structure")

    batch = 8
    f, g = Affine(3, 4), Affine(5, 6)
    params_f, params_g = f.init(generator=rng(61)), g.init(generator=rng(62))
    x_a = torch.randn(batch, 3, generator=rng(6))
    x_b = torch.randn(batch, 5, generator=rng(6))
    ybar_a = torch.randn(batch, 4, generator=rng(6))
    ybar_b = torch.randn(batch, 6, generator=rng(6))

    lhs = para_compose(para_prod(f, g), para_swap(f.cod, g.cod))
    rhs = para_compose(para_swap(f.dom, g.dom), para_prod(g, f))
    out_lhs, pb_lhs = lhs.run(((params_f, params_g), ()), (x_a, x_b))
    out_rhs, pb_rhs = rhs.run(((), (params_g, params_f)), (x_a, x_b))
    check("swap naturality fwd", out_lhs, out_rhs)

    pgrad_lhs, xgrad_lhs = pb_lhs((ybar_b, ybar_a))
    pgrad_rhs, xgrad_rhs = pb_rhs((ybar_b, ybar_a))
    (grad_f_lhs, grad_g_lhs), _ = pgrad_lhs
    _, (grad_g_rhs, grad_f_rhs) = pgrad_rhs
    check("swap nat pbar.f", grad_f_lhs, grad_f_rhs)
    check("swap nat pbar.g", grad_g_lhs, grad_g_rhs)
    check("swap nat xbar", xgrad_lhs, xgrad_rhs)

    swap_inv = compose(swap(5, 4), swap(4, 5))
    x_a = torch.randn(batch, 5, generator=rng(90))
    x_b = torch.randn(batch, 4, generator=rng(90))
    out, pb = swap_inv.run((x_a, x_b))
    check("swap>>swap fwd", out, (x_a, x_b))
    cotan_a = torch.randn(batch, 5, generator=rng(90))
    cotan_b = torch.randn(batch, 4, generator=rng(90))
    check("swap>>swap pb", pb((cotan_a, cotan_b)), (cotan_a, cotan_b))

    dim = 6
    x = torch.randn(12, dim, generator=rng(7))
    u = torch.randn(12, dim, generator=rng(7))
    v = torch.randn(12, dim, generator=rng(7))
    dup, add = lift(diagonal(dim)), lift(codiagonal(dim))

    out_dup, pb_dup = dup.run((), x)
    out_add, pb_add = add.run((), (u, v))
    check("dup fwd", out_dup, (x, x))
    check("add fwd", out_add, u + v)
    _, xgrad_dup = pb_dup((u, v))
    _, xgrad_add = pb_add(u)
    check("R[dup]=add", xgrad_dup, u + v)
    check("R[add]=dup", xgrad_add, (u, u))

    term = terminal(5)
    x_term = torch.randn(batch, 5, generator=rng(91))
    out_term, pb_term = term.run(x_term)
    assert out_term == (), f"Terminal fwd should give (), got {out_term}"
    print(f"  PASS: terminal fwd = ()")
    check("terminal pb", pb_term(()), torch.zeros(batch, 5))

    dim_a, dim_b = 5, 4
    x_a = torch.randn(batch, dim_a, generator=rng(80))
    x_b = torch.randn(batch, dim_b, generator=rng(80))
    cotan_a = torch.randn(batch, dim_a, generator=rng(80))
    cotan_b = torch.randn(batch, dim_b, generator=rng(80))

    out_p1, pb_p1 = proj1(dim_a, dim_b).run((x_a, x_b))
    out_p2, pb_p2 = proj2(dim_a, dim_b).run((x_a, x_b))
    check("pi1 fwd", out_p1, x_a)
    check("pi2 fwd", out_p2, x_b)
    grad_a1, grad_b1 = pb_p1(cotan_a)
    check("pi1 pb", (grad_a1, grad_b1), (cotan_a, torch.zeros(batch, dim_b)))
    grad_a2, grad_b2 = pb_p2(cotan_b)
    check("pi2 pb", (grad_a2, grad_b2), (torch.zeros(batch, dim_a), cotan_b))

    mor_f, _ = linear_mor(5, 3, gen=rng(81))
    mor_g, _ = linear_mor(5, 4, gen=rng(81))
    x_fan = torch.randn(5, generator=rng(81))
    out_fan, _ = fanout(mor_f, mor_g).run(x_fan)
    check("fanout left", out_fan[0], mor_f(x_fan))
    check("fanout right", out_fan[1], mor_g(x_fan))

    res_f, _ = linear_mor(6, 6, gen=rng(82))
    x_res = torch.randn(6, generator=rng(82))
    out_res, pb_res = residual(res_f).run(x_res)
    check("residual fwd", out_res, x_res + res_f(x_res))
    cotan_res = torch.randn(6, generator=rng(82))
    _, pb_inner = res_f.run(x_res)
    check("residual pb", pb_res(cotan_res), cotan_res + pb_inner(cotan_res))

    f, _ = linear_mor(5, 6, gen=rng(92))
    g, _ = linear_mor(4, 7, gen=rng(92))
    h, _ = linear_mor(3, 2, gen=rng(92))
    x_a = torch.randn(5, generator=rng(92))
    x_b = torch.randn(4, generator=rng(92))
    x_c = torch.randn(3, generator=rng(92))

    out_lhs, pb_lhs = prod(prod(f, g), h).run(((x_a, x_b), x_c))
    out_rhs, pb_rhs = prod(f, prod(g, h)).run((x_a, (x_b, x_c)))
    (fa_lhs, gb_lhs), hc_lhs = out_lhs
    fa_rhs, (gb_rhs, hc_rhs) = out_rhs
    check("prod assoc f(a)", fa_lhs, fa_rhs)
    check("prod assoc g(b)", gb_lhs, gb_rhs)
    check("prod assoc h(c)", hc_lhs, hc_rhs)

    cotan_a = torch.randn(6, generator=rng(92))
    cotan_b = torch.randn(7, generator=rng(92))
    cotan_c = torch.randn(2, generator=rng(92))
    xgrad_lhs = pb_lhs(((cotan_a, cotan_b), cotan_c))
    xgrad_rhs = pb_rhs((cotan_a, (cotan_b, cotan_c)))
    (ga_lhs, gb_lhs), gc_lhs = xgrad_lhs
    ga_rhs, (gb_rhs, gc_rhs) = xgrad_rhs
    check("prod assoc pb a", ga_lhs, ga_rhs)
    check("prod assoc pb b", gb_lhs, gb_rhs)
    check("prod assoc pb c", gc_lhs, gc_rhs)


def test_ccc():
    print("\n[TEST] CCC structure")
    m, n, k = 4, 3, 5

    hom = internal_hom(m, n)
    assert hom == m * n, f"[R^{m}, R^{n}] should be R^{m * n}, got R^{hom}"
    W = torch.randn(n, m, generator=rng(10))
    w_flat = W.view(n * m)
    x = torch.randn(m, generator=rng(10))
    out, _ = evaluate(m, n).run((w_flat, x))
    check("eval = matmul", out, W @ x)

    def f_run(inp):
        ctx, inp_a = inp
        out = inp_a @ W.t()

        def pb(ybar):
            return (torch.zeros_like(ctx), ybar @ W)

        return out, pb

    f = Mor((k, m), n, f_run)
    composed = compose(prod(curry(f, k, m, n), identity(m)), evaluate(m, n))
    x_ctx = torch.randn(k, generator=rng(10))
    x_inp = torch.randn(m, generator=rng(10))
    check("eval.(curry(f) x id) = f", composed((x_ctx, x_inp)), f((x_ctx, x_inp)))

    ctx_d, inp_d, out_d = 4, 3, 5
    W2 = torch.randn(out_d, inp_d, generator=rng(83))

    def f2_run(inp):
        ctx, inp_a = inp
        out = inp_a @ W2.t()

        def pb(ybar):
            return (torch.zeros_like(ctx), ybar @ W2)

        return out, pb

    f2 = Mor((ctx_d, inp_d), out_d, f2_run)
    roundtrip = uncurry(curry(f2, ctx_d, inp_d, out_d), ctx_d, inp_d, out_d)
    x_ctx = torch.randn(ctx_d, generator=rng(83))
    x_inp = torch.randn(inp_d, generator=rng(83))
    check("uncurry(curry(f)) = f", roundtrip((x_ctx, x_inp)), f2((x_ctx, x_inp)), tol=dict(rtol=1e-3, atol=1e-3))


def test_enrichment():
    print("\n[TEST] Vect-enrichment")
    m, n = 5, 3
    f, _ = linear_mor(m, n, gen=rng(20))
    g, _ = linear_mor(m, n, gen=rng(21))
    x = torch.randn(m, generator=rng(22))
    ybar = torch.randn(n, generator=rng(22))

    check("f + zero = f", add_mor(f, zero_mor(m, n))(x), f(x))
    check("f + g = g + f", add_mor(f, g)(x), add_mor(g, f)(x))

    alpha = 0.7
    check("a(f+g) = af+ag",
          scale_mor(alpha, add_mor(f, g))(x),
          add_mor(scale_mor(alpha, f), scale_mor(alpha, g))(x),
          tol=LIN)

    _, pb_sum = add_mor(f, g).run(x)
    _, pb_f = f.run(x)
    _, pb_g = g.run(x)
    check("R[f+g] pb", pb_sum(ybar), pb_f(ybar) + pb_g(ybar))

    _, pb_scaled = scale_mor(alpha, f).run(x)
    check("R[a*f] pb", pb_scaled(ybar), alpha * pb_f(ybar))


def test_differential():
    print("\n[TEST] Differential laws")

    f, _ = linear_mor(5, 3, gen=rng(30))
    x = torch.randn(5, generator=rng(30))
    tangent = torch.randn(5, generator=rng(30))
    cotangent = torch.randn(3, generator=rng(30))
    passed = check_jvp_vjp_duality(f, x, tangent, cotangent)
    if passed:
        print(f"  PASS: JVP-VJP duality")
    else:
        print(f"  FAIL: JVP-VJP duality")

    assert passed

    model = mlp(7, [9], 5, act="tanh")
    params = model.init(generator=rng(51))
    x = torch.randn(11, 7, generator=rng(50))
    _, pb = model.run(params, x)
    ybar1 = torch.randn(11, 5, generator=rng(50))
    ybar2 = torch.randn(11, 5, generator=rng(50))
    alpha = 0.37

    pgrad_sum, xgrad_sum = pb(ybar1 + ybar2)
    pgrad1, xgrad1 = pb(ybar1)
    pgrad2, xgrad2 = pb(ybar2)
    check("pb(y1+y2) pbar", pgrad_sum, data_add(pgrad1, pgrad2), tol=LIN)
    check("pb(y1+y2) xbar", xgrad_sum, xgrad1 + xgrad2, tol=LIN)
    pgrad_sc, xgrad_sc = pb(alpha * ybar1)
    check("pb(a*y1) pbar", pgrad_sc, data_scale(alpha, pgrad1), tol=LIN)
    check("pb(a*y1) xbar", xgrad_sc, alpha * xgrad1, tol=LIN)

    f_lin, _ = linear_mor(5, 3, gen=rng(84))
    x_lin = torch.randn(5, generator=rng(84))
    v1 = torch.randn(3, generator=rng(84))
    v2 = torch.randn(3, generator=rng(84))
    passed_lin = check_pullback_linearity(f_lin, x_lin, v1, v2, alpha=0.7)
    if passed_lin:
        print(f"  PASS: pullback linearity utility")
    else:
        print(f"  FAIL: pullback linearity utility")

    assert passed_lin

    f_chain, _ = linear_mor(5, 4, gen=rng(85))
    g_chain, _ = linear_mor(4, 3, gen=rng(85))
    x_chain = torch.randn(5, generator=rng(85))
    zbar = torch.randn(3, generator=rng(85))
    passed_chain = check_chain_rule(f_chain, g_chain, x_chain, zbar)
    if passed_chain:
        print(f"  PASS: chain rule utility")
    else:
        print(f"  FAIL: chain rule utility")

    assert passed_chain


def test_vjp_vs_autograd():
    print("\n[TEST] VJP vs autograd")

    def check_model(name, model, params, x, ybar):
        _, pb = model.run(params, x)
        pgrad, xgrad = pb(ybar)
        _, pgrad_ref, xgrad_ref = autograd_vjp(model, params, x, ybar)
        check(f"{name} pbar", pgrad, pgrad_ref, tol=AD)
        check(f"{name} xbar", xgrad, xgrad_ref, tol=AD)

    affine = Affine(7, 5)
    check_model("affine", affine, affine.init(generator=rng(41)), torch.randn(16, 7, generator=rng(40)), torch.randn(16, 5, generator=rng(40)))

    mlp_model = mlp(7, [11, 13], 5, act="tanh")
    mlp_params = mlp_model.init(generator=rng(43))
    mlp_x = torch.randn(16, 7, generator=rng(42))
    labels = torch.randint(0, 5, (16,), generator=rng(42))
    loss_fn = SoftmaxCrossEntropy()
    logits, mlp_pb = mlp_model.run(mlp_params, mlp_x)
    _, dlogits = loss_fn.loss_and_grad(logits, labels)
    mlp_pgrad, mlp_xgrad = mlp_pb(dlogits)

    params_ref = tree_map(lambda t: t.detach().clone().requires_grad_(True), mlp_params)
    x_ref = mlp_x.detach().clone().requires_grad_(True)
    loss_ref, _ = loss_fn.loss_and_grad(mlp_model(params_ref, x_ref), labels)
    loss_ref.backward()
    check("mlp pbar", mlp_pgrad, tree_map(lambda t: t.grad.detach(), params_ref), tol=AD)
    check("mlp xbar", mlp_xgrad, x_ref.grad.detach(), tol=AD)

    res_model = residual_mlp(9, depth=2, act="softplus")
    res_params = res_model.init(generator=rng(45))
    res_x = torch.randn(8, 9, generator=rng(44))
    res_ybar = torch.randn(8, 9, generator=rng(44))
    res_out, res_pb = res_model.run(res_params, res_x)
    res_pgrad, res_xgrad = res_pb(res_ybar)
    res_out_ref, res_pgrad_ref, res_xgrad_ref = autograd_vjp(res_model, res_params, res_x, res_ybar)
    check("resmlp fwd", res_out, res_out_ref)
    check("resmlp pbar", res_pgrad, res_pgrad_ref, tol=AD)
    check("resmlp xbar", res_xgrad, res_xgrad_ref, tol=AD)

    conv = Conv2d(2, 3, 8, 8, kernel=3, padding=1)
    check_model("conv2d", conv, conv.init(generator=rng(47)), torch.randn(4, 2 * 8 * 8, generator=rng(46)), torch.randn(4, 3 * 8 * 8, generator=rng(46)))


def test_para():
    print("\n[TEST] Para laws")
    batch, dim = 8, 5

    par_id = para_identity(dim)
    x = torch.randn(batch, dim, generator=rng(60))
    out, pb = par_id.run((), x)
    check("para id fwd", out, x)
    ybar = torch.randn(batch, dim, generator=rng(60))
    pgrad, xgrad = pb(ybar)
    assert pgrad == (), "Para id param grads should be ()"
    check("para id xbar", xgrad, ybar)

    dim_a, dim_b, dim_c = 5, 6, 4
    f, g = Affine(dim_a, dim_b), Affine(dim_b, dim_c)
    params_f, params_g = f.init(generator=rng(71)), g.init(generator=rng(72))
    x = torch.randn(batch, dim_a, generator=rng(70))

    out_composed = to_mor(para_compose(f, g), (params_f, params_g))(x)
    out_separate = compose(to_mor(f, params_f), to_mor(g, params_g))(x)
    check("forgetful functor", out_composed, out_separate)


def test_lenses():
    print("\n[TEST] Lens laws")

    id_lens = lens_id()
    x = torch.randn(5)
    val, res = id_lens.get(x)
    check("lens id roundtrip", id_lens.put(res, val), x)

    lens1 = Lens(get=lambda s: (s * 2, s), put=lambda r, a: r + a)
    lens2 = Lens(get=lambda s: (s + 1, s), put=lambda r, a: r * a)
    composed = lens_compose(lens1, lens2)
    state = torch.tensor([3.0])
    val, res = composed.get(state)
    check("lens compose get", val, torch.tensor([7.0]))
    check("lens compose put", composed.put(res, torch.tensor([0.5])), torch.tensor([6.0]))

    prod_lens = lens_prod(lens1, lens2)
    state1, state2 = torch.tensor([3.0]), torch.tensor([5.0])
    (val_a, val_b), res = prod_lens.get((state1, state2))
    check("lens prod get", (val_a, val_b), (torch.tensor([6.0]), torch.tensor([6.0])))
    put_result = prod_lens.put(res, (torch.tensor([1.0]), torch.tensor([2.0])))
    check("lens prod put", put_result, (torch.tensor([4.0]), torch.tensor([10.0])))

    model = mlp(7, [11], 5, act="tanh")
    params = model.init(generator=rng(87))
    x = torch.randn(16, 7, generator=rng(86))
    labels = torch.randint(0, 5, (16,), generator=rng(86))
    loss_fn = SoftmaxCrossEntropy()
    opt = AdamW(weight_decay=0.01)
    lr = 1e-3

    params_lens, params_imp = copy.deepcopy(params), copy.deepcopy(params)
    state_lens, state_imp = opt.init_state(params_lens), opt.init_state(params_imp)

    out_lens, loss_lens, _ = training_step(model, loss_fn, opt, x, labels, params_lens, state_lens, lr)

    logits, pb = model.run(params_imp, x)
    loss_imp, dlogits = loss_fn.loss_and_grad(logits, labels)
    grads_imp, _ = pb(dlogits)
    out_imp, _ = opt.step(params_imp, state_imp, grads_imp, lr)

    check("train step loss", loss_lens, loss_imp)
    check("train step params", out_lens, out_imp)


def main():
    torch.set_printoptions(precision=4, sci_mode=True)
    print("CatDL - Categorical Deep Learning Law Verification")

    test_crdc()
    test_cartesian()
    test_ccc()
    test_enrichment()
    test_differential()
    test_vjp_vs_autograd()
    test_para()
    test_lenses()

    print("\nAll tests passed")


if __name__ == "__main__":
    main()
