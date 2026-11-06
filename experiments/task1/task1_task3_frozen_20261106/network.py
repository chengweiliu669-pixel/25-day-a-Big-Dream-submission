"""Official Task3 parameter resolution."""
import numpy as np
import pandas as pd
from pathlib import Path


def network_parameters(panel_dir: Path):
    """Per-link triangular FD parameters, taken from one source at a time.

    This used to read lanes/free_speed/capacity from links.csv and only
    critical_density/k_jam from fd_parameters.csv. The two describe different
    worlds. links.csv comes from an OpenStreetMap extract: free_speed is a flat
    105.0 default on every non-placeholder link in all ten panels, and capacity
    is the formula lanes*2000 over map lane tags that disagree with the detector
    station metadata on 461 of 993 rows. fd_parameters.csv is measured -
    free_speed is the q0.99 of observed speed, capacity the q0.999 of observed
    flow, and critical_density is capacity/free_speed by construction.

    Mixing them broke the triangle the scorer then evaluates. On the 856 links
    Task 3 actually scores, v_f * k_crit differed from capacity on 851, by a
    median of 25-37%: the scorer read v_f = 105.0 where the detectors measure
    113-117.5, and capacity 8000-12000 where they measure 6324-9378.

    fd_parameters is now authoritative for all five parameters wherever it
    covers a link. It is keyed by station and several stations can share a link,
    so capacity and free_speed are averaged over the link's stations and
    critical_density is recomputed from the averages - averaging it separately
    would reintroduce the same inconsistency, since mean(v_f) * mean(k_crit) is
    not mean(capacity). length_km has no counterpart there and stays with
    links.csv; it never enters the FD or conservation maths, only the dropna
    guard below.
    """
    n = pd.read_csv(panel_dir / "network" / "links.csv")
    n.link_id = n.link_id.astype(str); n = n.drop_duplicates("link_id")
    keep = ["link_id","length_km","free_speed_kmh","capacity_vph","lanes"]
    out = n[keep].copy()
    for c, default in (("length_km",1.0),("free_speed_kmh",100.0)):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default)
    out["lanes"] = pd.to_numeric(out["lanes"], errors="coerce").fillna(1.0).clip(lower=1.0)
    out.capacity_vph = pd.to_numeric(out.capacity_vph, errors="coerce")
    lanes = pd.to_numeric(n.get("lanes", pd.Series(4.0, index=n.index)), errors="coerce").fillna(4.0)
    lane_fallback = pd.Series(1800.0 * lanes.to_numpy(), index=n.index)
    lane_fallback.index = out.index
    out.capacity_vph = out.capacity_vph.fillna(lane_fallback)

    fd = panel_dir / "network" / "fd_parameters.csv"
    if fd.exists():
        f = pd.read_csv(fd); f.link_id = f.link_id.astype(str)
        have = [c for c in ("free_speed_kmh","capacity_vph","lanes","k_jam") if c in f.columns]
        m = f.groupby("link_id", as_index=False)[have].mean()
        out = out.merge(m, on="link_id", how="left", suffixes=("", "_fd"))
        for c in ("free_speed_kmh","capacity_vph","lanes"):
            if f"{c}_fd" in out.columns:
                out[c] = out[f"{c}_fd"].fillna(out[c])
                out = out.drop(columns=[f"{c}_fd"])
        if "k_jam" not in out.columns:
            out["k_jam"] = np.nan
        # Recomputed, never averaged: this is what keeps v_f * k_crit == capacity.
        out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    else:
        out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
        out["k_jam"] = np.nan
    # A link with no measured jam density falls back to 100 veh/km/lane, the
    # assumption the released diagrams already use, rather than 2*k_crit - that
    # older fallback forced cap/(k_jam-k_crit) to equal the free-flow speed,
    # putting jam waves at 105 km/h instead of the 15-20 a freeway shows.
    out["k_jam"] = out.k_jam.fillna(100.0 * out.lanes.clip(lower=1.0))
    out["k_jam"] = np.maximum(out.k_jam, out.critical_density * 1.05)
    return out
