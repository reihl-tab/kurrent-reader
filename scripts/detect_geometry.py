#!/usr/bin/env python3
"""Adaptive pencil-rule detection for Eschau-1081 table spreads (DRAFT, unverified).

Per page: find gutter (min dark-pixel column in center band), then per half try
darkness thresholds [5..20], pick the one whose detected v/h rule counts are
closest to expected table structure. Morphology open with long thin kernels
extracts faint pencil rules that global thresholds miss (rules sit only ~5-10
gray levels below background).

KNOWN LIMITATIONS (verified 2026-10-26):
- Layouts vary per page; g13 is prose-with-year-headers, not a ruled table.
- v_rules are plausible leads, h_rules are unreliable (miss rows / cut text).
- Every page needs visual verification against overlays/ before use.
"""
import cv2, numpy as np, json, glob, os, sys

def groups(idx, gap=12):
    gs = []
    for i in idx:
        if gs and i - gs[-1][-1] <= gap: gs[-1].append(i)
        else: gs.append([i])
    return [int(np.mean(g)) for g in gs if len(g) >= 2]

def detect_half(half, x0):
    H, hW = half.shape
    blur = cv2.GaussianBlur(half, (0,0), 3)
    diff = cv2.subtract(blur, half)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    best = None
    for thr in [5, 6, 8, 10, 12, 15, 20]:
        _, bw = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
        hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
        colprof = (vl > 0).sum(axis=0)
        rowprof = (hl > 0).sum(axis=1)
        v = [x + x0 for x in groups(list(np.where(colprof > 0.10*H)[0]))]
        h = groups(list(np.where(rowprof > 0.10*hW)[0]))
        score = -abs(len(v) - 6)*100 - abs(len(h) - 12)*10 + colprof.sum()/1e6
        if best is None or score > best[0]:
            best = (score, thr, v, h)
    return best[1], best[2], best[3]

def detect_page(img):
    H, W = img.shape
    dark = (img < 150).astype(np.uint8)
    prof = dark.mean(axis=0)
    c0, c1 = int(W*0.42), int(W*0.58)
    gut = c0 + int(np.argmin(prof[c0:c1]))
    halves = []
    for (x0, x1) in [(0, gut), (gut, W)]:
        thr, v, h = detect_half(img[:, x0:x1], x0)
        halves.append({'x_range': [x0, x1], 'thr': thr, 'v_rules': v, 'h_rules': h})
    return gut, halves

if __name__ == '__main__':
    out = {}
    for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv)>1 else '/tmp/pages', '*.jpg'))):
        name = os.path.basename(f)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        gut, halves = detect_page(img)
        out[name] = {'size': list(img.shape[::-1]), 'gutter': gut, 'halves': halves}
        print(name, gut, [(h['thr'], len(h['v_rules']), len(h['h_rules'])) for h in halves])
    json.dump(out, open(sys.argv[2] if len(sys.argv)>2 else '/tmp/geometry_halves_adaptive.json', 'w'), indent=1)
