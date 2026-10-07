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
