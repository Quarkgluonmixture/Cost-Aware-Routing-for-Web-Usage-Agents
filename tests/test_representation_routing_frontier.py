"""The fixed frontier must be the mixture envelope, not the six corners.

Catches: a flipped cross-product sign (the hull keeps a point below the chord, or drops one
above it), and a frontier that keeps falling after the best-SR mode instead of staying flat
(free disposal: paying more never forces a lower SR).
"""
from scripts.analysis.representation_routing_frontier import fixed_hull, hull_sr_at


def test_point_below_chord_is_not_on_frontier_and_chord_is_interpolated():
    # B sits below the A–C chord (chord SR at cost 2 is 20); D is dearer and worse than C.
    pts = [(1.0, 10.0), (2.0, 15.0), (3.0, 30.0), (4.0, 5.0)]
    assert fixed_hull(pts) == [(1.0, 10.0), (3.0, 30.0)]
    h = fixed_hull(pts)
    assert hull_sr_at(h, 2.0) == 20.0          # 50/50 mixture of A and C
    assert hull_sr_at(h, 4.0) == 30.0          # flat after the best-SR mode, not D's 5
    assert hull_sr_at(h, 0.5) == 10.0          # cheaper than every mode: cheapest mode's SR


def test_point_above_chord_stays_on_frontier():
    pts = [(1.0, 10.0), (2.0, 25.0), (3.0, 30.0)]
    assert fixed_hull(pts) == pts


def test_frontier_gain_on_normalised_budget():
    """Catches: a gain that is not zero when the curve adds nothing, a budget axis anchored on
    the hull instead of the cheapest/dearest fixed mode, and a curve point cheaper than every
    fixed mode being dropped instead of lifting the low-budget end."""
    from scripts.analysis.representation_routing_frontier import U_GRID, frontier_gain

    fixed = [(1.0, 10.0), (3.0, 30.0), (5.0, 20.0)]      # u = 0 at cost 1, u = 1 at cost 5
    at = {u: i for i, u in enumerate(U_GRID)}

    below = frontier_gain(fixed, [(2.0, 15.0)])            # under the 10→30 chord (20 at cost 2)
    assert abs(below).max() < 1e-9

    above = frontier_gain(fixed, [(2.0, 26.0)])
    assert abs(above[at[0.25]] - 6.0) < 1e-9               # cost 2: 26 vs chord 20
    assert abs(above[at[0.0]]) < 1e-9 and abs(above[at[0.5]]) < 1e-9 and abs(above[at[1.0]]) < 1e-9

    cheaper = frontier_gain(fixed, [(0.5, 12.0)])
    assert abs(cheaper[at[0.0]] - 5.6) < 1e-9              # chord 0.5→3 at cost 1 = 15.6 vs 10
