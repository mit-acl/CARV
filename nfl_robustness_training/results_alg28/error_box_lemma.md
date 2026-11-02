# The certified-set error box

What the lookahead needs, and why the deployed bound was loose by a factor that
grows with the horizon.

## Setting

State `x = (p_x, p_y, θ) ∈ R^3`, unicycle at constant speed `V`, step `dt`:

    f(x, u) = ( p_x + dt·V·cos θ ,  p_y + dt·V·sin θ ,  θ + dt·u ).

The filter never sees `x`. At each executed step `t` it holds a **certified
set** `S_t` — a box guaranteed to contain the true state — and applies the
control the network computes at that box's centre,

    c_t = centre(S_t),    u_t = π(c_t),    e_t := c_t − x_t,

so the estimation error obeys `|e_t| ≤ hw(S_t)` componentwise, where `hw`
denotes the half-width vector of a box. The certified set is maintained by

    S_{t+1} = F(S_t, u_t) ∩ M_{t+1},     M_{t+1} := [z_{t+1} − v, z_{t+1} + v],

with `F(S,u)` the interval image of `S` under `f(·,u)` and `M` the measurement
box for sensor bound `v = vmax` (componentwise).

The lookahead propagates a **reachable box** `X_k ⊇ {x_{t+k}}` under the
estimate-feedback law, and to do so it must supply an **error box** `E_k ⊇
{e_{t+k}}`. The deployed rule was

    (X)   E_k = [−ε_k, ε_k],   ε_k = min(v, hw(X_k)).                 [alg28_sound.py]

This is sound — `x̂ ∈ S_k ⊆ X_k` gives `|e_k| ≤ hw(X_k)`, and the sensor gives
`|e_k| ≤ v` — but `X_k` grows along the chain, so within a few steps `ε_k`
saturates at the raw sensor bound `v` in every component and stays there. The
lemma below replaces `hw(X_k)` with a bound on `hw(S_k)`, which does **not**
grow the same way.

## Lemma (certified-set error recursion)

Let `S ⊂ R^3` be a box and let `u ∈ R` be **any** scalar. Then

    (i)   hw_θ ( F(S,u) )  =  hw_θ(S),
    (ii)  hw_p ( F(S,u) )  ≤  hw_p(S) + dt·V·hw_θ(S)      for p ∈ {x, y},
    (iii) hw   ( F(S,u) ∩ M )  ≤  min( v, hw(F(S,u)) )     componentwise.

Consequently, writing `h_k` for the bound on `hw(S_{t+k})` produced by

    h_0 = hw(S_t),
    h_{k+1} = min( v,  ( h_k^x + dt·V·h_k^θ ,  h_k^y + dt·V·h_k^θ ,  h_k^θ ) ),

we have `hw(S_{t+k}) ≤ h_k` and therefore the tighter, still sound rule

    (S)   ε_k = min( v, hw(X_k), h_k ).

**Proof.** (i) The heading update `θ ↦ θ + dt·u` is a translation of `R`; `u`
does not depend on the point of `S`, so the image of the θ-interval is an
interval of the same width. (ii) For `θ_1, θ_2` in an interval of half-width
`h`, `|cos θ_1 − cos θ_2| ≤ |θ_1 − θ_2| ≤ 2h`, so the interval hull of
`dt·V·cos θ` over `S` has half-width at most `dt·V·h`; adding the independent
`p_x`-interval gives the claim, and identically for `sin`. (iii) The
intersection of two boxes is a box whose half-width is componentwise at most
the smaller of the two. ∎

The recursion is **uniform in `u`**: (i) and (ii) hold for every scalar `u`, so
`h_k` can be computed at planning time without knowing the controls the network
will produce. It is also **independent of the measurement realisation**: (iii)
bounds the intersection by `v` whatever `z` turns out to be. Those two
properties are what make it usable in the lookahead rather than only in
hindsight.

## Structural conditions

The lemma is not generic; it buys its tightness from five specific facts. Each
is a condition a reader should be able to check, and each fails audibly.

**C1 — one scalar control per step, applied open-loop to the whole set.** The
filter computes `u_t` once, from the centre, and applies it to every point of
`S_t`. If the set were propagated under a *range* of controls `[u̲, ū]` — which
is what the decoupled RSOA does — (i) becomes `hw_θ(S) + dt·(ū − u̲)/2` and the
heading width grows. This is the single structural fact the whole result rests
on, and it is a property of the *executed* loop, not of the relaxation.

**C2 — heading enters its own update only additively.** `θ' = θ + dt·u` with
`u` independent of the point. Any `θ`-dependence inside the propagated set
(a rate that depends on heading, a nonholonomic correction) turns (i) into an
inequality with a Lipschitz factor and the recursion starts to grow.

**C3 — constant speed, and position Lipschitz-1 in heading.** `|∂p'/∂θ| ≤
dt·V` uses `|sin|, |cos| ≤ 1` and `V` constant. With `V` a control or state,
replace `V` by `sup V` over the set and add `dt·hw_V` to (ii).

**C4 — heading is carried unwrapped.** (i) is an isometry statement about `R`.
Normalising `θ` to `(−π, π]` would break the box representation across the
branch cut. The implementation keeps `θ` unwrapped; if that changes, (i) needs
a wrap-aware interval type.

**C5 — the measurement box really contains the truth, every step.** `|z_t −
x_t| ≤ v` componentwise. Under a hard sensor bound (`TTTCARV_NOISE=uniform`,
`v = 3σ` with draws from `U(−v, v)`) this holds by construction. Under Gaussian
noise it holds only with probability, per component per step, and the first
violation invalidates every `h_k` downstream — this is the assumption the
Gaussian stress test exists to attack.

**C6 — the intersection is non-empty.** If `F(S_k,u_k) ∩ M_{k+1} = ∅` then C5
was violated. The implementation's REPAIR path resets `S` to the measurement
box, half-width `v`, which is sound but breaks the chain that `h_k` was built
on. Rule (S) must therefore fall back to rule (X) for the remainder of any
trial in which a repair fires. (Measured: 0 repairs in 12 000 batch steps under
C5, so this path is guarded but untested by experiment.)

## Why only the joint RSOA can use it

Mode B (decoupled) bounds the network over the whole reachable box: `U ⊇
π(X_k ⊕ E_k)`, then propagates `X_k` under the interval `U`. Its soundness
argument is `x̂_k ∈ S_k ⊆ X_k`, so any `u` the network can produce is in
`π(X_k)`. A bound on `hw(S_k)` says how *wide* the certified set is, not
*where in `X_k` it sits* — so it cannot shrink the network's input box, and
rule (S) buys nothing.

Mode J (joint) bounds `F(x, e) ↦ f(x, π(x + e))` over the product `X_k × E_k`
as one graph. Here `e` is an explicit input with its own box, and the only
thing required of `E_k` is `e_k ∈ E_k`. That is exactly what the lemma
delivers, so the tightening applies directly.

## Measured effect

On real certified sets (seed 1443680074, `t = 36`, ten steps ahead), rule (S)
against rule (X): heading `0.892 → 0.279` rad, x/y `0.616 → 0.297` m, at
identical cost. In closed loop (200 paired seeds per law, all three arms of a law
sequential on one exclusive node), rule (S) removes 85–113 safety-filter
interventions under every noise law tested, winning ~4:1 on seeds, with no
change in the per-step compute budget. As a share of the excess over the
true-state floor that is 61 % under uniform noise (710 → 619, floor 561) but
only 22 % under a strongly peaked truncated Gaussian (1164 → 1079, floor 769):
the absolute benefit is distribution-robust, the percentage is not.
(An earlier 49 % figure came from arms run on separate nodes and is superseded.)
