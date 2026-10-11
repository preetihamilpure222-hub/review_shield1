"""Stylometric features shared by training and prediction."""
import re
import numpy as np

FP = set("i me my mine myself we our us".split())
GEN = set("all always never every everyone nobody best worst perfect amazing awesome ever".split())


def feats(raw):
    t = str(raw); w = t.split(); n = max(len(w), 1)
    lw = [x.lower().strip(".,!?;:\"'()") for x in w]
    return [len(t), n, t.count("!") / n, t.count("?") / n,
            sum(c.isupper() for c in t) / max(len(t), 1),
            sum(x in FP for x in lw) / n, sum(x in GEN for x in lw) / n,
            len(set(lw)) / n, float(np.mean([len(x) for x in lw])) if lw else 0.0,
            sum(not c.isalnum() and c != " " for c in t) / max(len(t), 1),
            len(re.split(r"[.!?]+", t)), sum(c.isdigit() for c in t) / max(len(t), 1)]
