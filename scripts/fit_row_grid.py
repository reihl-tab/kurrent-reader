#!/usr/bin/env python3
"""Periodic row-grid fitter for Eschau-1081 table spreads (DRAFT, per-page verification required).

Rows in these registers are evenly spaced entries (1-3 handwriting lines each).
Detected h_rules (morphological open, threshold sweep) are a noisy SUBSET of the
true rules, so we RANSAC a uniform grid: for step 80..220 and every offset, count
detected rules within 10px of a grid line; among grids within 1 hit of the max,
pick the LARGEST step (avoids oversplitting multi-line entries - step 96 beat the
true 196 on s04 by spurious hits until this rule), tie-break by low ink at lines.

VALIDATED 2026-10-27: s04 left half grid [8..2164 step 196] puts the held-out
Gerson Mosbacher band (truth y995-1165) fully inside row 988-1184. Visual check
passed on s02 (ruled, 164-204) and g04 (one-line entries, 116). Pages with few
detected rules (g10, s05, s08, g07, g13-prose) are LOW CONFIDENCE - verify per
page against rowgrid_overlays/ before using for crops.
"""
import cv2, numpy as np, json, glob, os, sys

def groups(idx, gap=12):
    gs = []
    for i in idx:
        if gs and i - gs[-1][-1] <= gap: gs[-1].append(i)
        else: gs.append([i])
    return [int(np.mean(g)) for g in gs if len(g) >= 2]

def detect_h(half, w):
    blur = cv2.GaussianBlur(half, (0,0), 3)
    diff = cv2.subtract(blur, half)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    best = None
    for thr in [5, 6, 8, 10, 12, 15, 20]:
        _, bw = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
        rowprof = (hl > 0).sum(axis=1)
        h = groups(list(np.where(rowprof > 0.10*w)[0]))
        if best is None or abs(len(h)-12) < abs(len(best)-12): best = h
    return best

def fit_grid(rules, inkprof, H, tol=10):
    cands = []
    for step in range(80, 221):
        for off in range(0, step):
            grid = np.arange(off, H, step)
            if len(grid) < 3: continue
            hits = sum(1 for y in rules if np.min(np.abs(grid - y)) <= tol)
            ink_pen = float(np.mean([inkprof[max(y-8,0):min(y+9,H)].max() for y in grid]))
            cands.append((hits, step, off, ink_pen))
    mx = max(c[0] for c in cands)
    top = [c for c in cands if c[0] >= mx - 1]
    top.sort(key=lambda c: (-c[1], c[3]))
    hits, step, off, _ = top[0]
    return [int(y) for y in np.arange(off, H, step)], hits, len(rules), step

def fit_page(img):
    H, W = img.shape
    dark = (img < 150).astype(np.uint8)
    prof0 = dark.mean(axis=0)
    gut = int(W*0.42) + int(np.argmin(prof0[int(W*0.42):int(W*0.58)]))
    halves = []
    for (x0, x1) in [(0, gut), (gut, W)]:
        half = img[:, x0:x1]
        blur = cv2.GaussianBlur(half, (0,0), 3)
        diff = cv2.subtract(blur, half)
        ink = (diff > 20).astype(np.float32)
        inkprof = np.convolve(ink.mean(axis=1), np.ones(9)/9, mode='same')
        inkprof /= (inkprof.max() + 1e-9)
        rules = detect_h(half, x1-x0)
        grid, hits, n, step = fit_grid(rules, inkprof, H)
        rows = [[int(grid[i]), int(grid[i+1])] for i in range(len(grid)-1)]
        halves.append({'x_range': [x0, x1], 'h_detected': rules, 'grid': grid,
                       'rows': rows, 'fit': f'{hits}/{n}', 'step': step})
    return gut, halves

if __name__ == '__main__':
    out = {}
    for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv)>1 else '/tmp/pages', '*.jpg'))):
        name = os.path.basename(f)
        gut, halves = fit_page(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
        out[name] = {'gutter': gut, 'halves': halves}
        print(name, [(h['fit'], h['step'], len(h['rows'])) for h in halves])
    json.dump(out, open(sys.argv[2] if len(sys.argv)>2 else '/tmp/row_grids.json', 'w'), indent=1)
