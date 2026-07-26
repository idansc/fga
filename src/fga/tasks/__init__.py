"""Applications built on top of [`fga.attention`].

Each task package wires the general factor-graph attention to a concrete problem:
how its modalities become utilities, what the priors are, and how the attended
representations are scored. `visual_dialog` is the reference implementation, and
the one the paper reports.
"""
