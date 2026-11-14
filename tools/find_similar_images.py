#!/usr/bin/env python3
"""Find (and optionally remove) near-duplicate sonar images via SSIM/MSE/PSNR.

A forward-looking sonar dataset often holds frames that are almost identical
(the sonar barely moved between captures). If such near-duplicates end up split
across train/val/test, the model is tested on frames it effectively saw during
training -- data leakage. This tool locates near-duplicate images so you can
drop the redundant ones before splitting.

How it works
------------
Every image is loaded as grayscale and resized to a small common canvas
(``--size`` x ``--size``) so differently-sized frames are comparable and the
pairwise pass stays cheap. For each pair of images three similarity metrics are
computed:

  * SSIM  -- structural similarity, in [-1, 1]; 1.0 == identical. Robust to
    small intensity shifts, so it is the primary signal for "same scene".
  * MSE   -- mean squared error on [0, 1]-scaled pixels; 0.0 == identical.
  * PSNR  -- peak signal-to-noise ratio in dB; higher == more similar
    (inf == identical).

A pair counts as a pixel-level near-duplicate when the enabled criteria agree.
With ``--require all`` (default) a pair must satisfy every enabled metric's
threshold; with ``--require any`` satisfying one is enough. Disable a metric by
passing its threshold as empty, e.g. ``--mse ''``.

Annotation gate (``--ann-dir``)
-------------------------------
Sonar images carry heavy reverberation and speckle noise, so SSIM/MSE/PSNR can
all land within threshold for two frames that are visually different scenes --
a false "same image". When a folder of per-image XML annotations is supplied,
each pixel-similar pair must also be CONFIRMED by its annotations before it is
treated as a duplicate:

  * The two images must share AT LEAST ONE object class. (We do not require the
    full class sets to match -- a genuine duplicate may have one annotation that
    missed an object.)
  * For some shared class, a box in image A must overlap a box of that class in
    image B by ``--ann-overlap`` or more (boxes are normalized to [0, 1] by
    each image's own size, so position -- not just presence -- is what counts).

This cuts both error directions: two different frames that merely contain the
same class have boxes in different places and are rejected; two truly identical
frames are confirmed by the overlapping box of a shared class. Pairs the pixel
metrics called duplicate but the annotations overruled are logged separately.
Use ``--ann-missing`` to choose what happens when an image has no XML.

Near-duplicate pairs are merged into groups (union-find). Within each group the
earliest image in natural frame order is kept as the representative and the rest
are flagged as duplicates to remove.

Speed
-----
The comparison is O(n^2) in the number of images. ``--size`` keeps each metric
cheap; a cheap MSE pre-gate (``--prefilter-mse``) skips the costlier SSIM for
pairs that are already obviously different. For very large folders compare on a
small ``--size`` (e.g. 64) first.

Outputs
-------
  * ``duplicates.csv``   -- one row per near-duplicate pair with all metrics
    and (when the gate is on) the annotation verdict and matching class.
  * ``rejected_by_annotation.csv`` -- pixel-similar pairs the annotation gate
    vetoed (useful for tuning the thresholds).
  * ``groups.csv``       -- group id, representative, duplicate members.
  * ``remove_list.txt``  -- the duplicate image paths to delete (survivors omitted).

With ``--action`` the duplicates can also be moved aside, symlinked into a
"deduped" folder of survivors, or deleted. Whenever ``--ann-dir`` and/or
``--mask-dir`` are set, each image's matching XML annotation and segmentation
mask (same file stem) are deleted / moved / symlinked alongside it, so the
annotation and mask sets stay in sync with the images.

Interactive review (``--review``)
---------------------------------
For a quick human check, ``--review`` opens a window after processing: the
group representative (always kept) on the left, each other member of the group
on the right one at a time, with **Keep** and **Delete** buttons (keys ``k`` /
``d``). Delete removes the image in place immediately -- together with its
matching XML (``--ann-dir``) and mask (``--mask-dir``) when set -- while Keep
leaves it. When a
group is finished the next group starts automatically; close the window to
stop. ``--review`` overrides ``--action`` and needs a GUI-capable matplotlib.

Examples
--------
    python tools/find_similar_images.py --img-dir /path/to/Images
    python tools/find_similar_images.py --img-dir Images --ssim 0.95 --size 128
    # with the annotation gate (recommended for noisy sonar):
    python tools/find_similar_images.py --img-dir Images \\
        --ann-dir BoxAnnotations --ssim 0.9 --ann-overlap 0.3
    # human-check each group and delete in place:
    python tools/find_similar_images.py --img-dir Images \\
        --ann-dir BoxAnnotations --review
    python tools/find_similar_images.py --img-dir Images --require any \\
        --ssim 0.97 --psnr 32 --action move --out-dir dedup_out
"""
import argparse
import csv
import os
import os.path as osp
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from glob import glob

import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        'scikit-image is required: pip install scikit-image') from e

try:
    import cv2

    def _imread_gray(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def _resize(img, size):
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    def _imread_display(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
except ImportError:  # fall back to PIL if OpenCV is absent
    from PIL import Image

    def _imread_gray(path):
        try:
            return np.asarray(Image.open(path).convert('L'))
        except Exception:
            return None

    def _resize(img, size):
        return np.asarray(
            Image.fromarray(img).resize((size, size), Image.BILINEAR))

    def _imread_display(path):
        try:
            return np.asarray(Image.open(path).convert('RGB'))
        except Exception:
            return None


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def opt_float(v):
    """argparse type: '' / 'none' disables a metric, else float."""
    if v is None or v.strip().lower() in ('', 'none', 'off'):
        return None
    return float(v)


def parse_args():
    p = argparse.ArgumentParser(
        description='Find near-duplicate images via SSIM, MSE and PSNR.')
    p.add_argument('--img-dir', required=True,
                   help='folder of images to scan (non-recursive)')
    p.add_argument('--recursive', action='store_true',
                   help='also scan sub-folders of --img-dir')
    p.add_argument('--out-dir', default=None,
                   help='where to write reports (default: <img-dir>/_dedup)')
    p.add_argument('--size', type=int, default=128,
                   help='resize both images to size x size for comparison')
    p.add_argument('--ssim', type=opt_float, default=0.95,
                   help="SSIM >= this means duplicate ('' to disable)")
    p.add_argument('--mse', type=opt_float, default=0.01,
                   help="MSE <= this means duplicate, on [0,1] pixels "
                        "('' to disable)")
    p.add_argument('--psnr', type=opt_float, default=30.0,
                   help="PSNR(dB) >= this means duplicate ('' to disable)")
    p.add_argument('--require', choices=['all', 'any'], default='all',
                   help='duplicate if ALL enabled metrics agree (default) '
                        'or if ANY one does')
    p.add_argument('--prefilter-mse', type=opt_float, default=0.05,
                   help='skip SSIM/PSNR for pairs whose MSE exceeds this '
                        '(cheap gate; set empty to always compute all)')
    # --- annotation gate (guards against noise-fooled pixel metrics) ---
    p.add_argument('--ann-dir', default=None,
                   help='folder of per-image XML annotations (matched by file '
                        'stem). When given, a pixel-similar pair is only kept '
                        'as a duplicate if the annotations also confirm it.')
    p.add_argument('--ann-overlap', type=float, default=0.3,
                   help='two boxes of a shared class count as the same object '
                        'when their overlap >= this (normalized coords)')
    p.add_argument('--ann-overlap-metric', choices=['coverage', 'iou'],
                   default='coverage',
                   help='coverage = intersection / smaller box (lenient, '
                        'default); iou = intersection / union')
    p.add_argument('--ann-missing', choices=['pixel-only', 'skip'],
                   default='pixel-only',
                   help='when a pair lacks annotations: fall back to the '
                        'pixel decision (default) or never flag it (skip)')
    p.add_argument('--mask-dir', default=None,
                   help='folder of per-image segmentation masks (matched by '
                        'file stem). When given, each image\'s mask is '
                        'deleted / moved / symlinked alongside the image.')
    p.add_argument('--action', choices=['report', 'move', 'symlink', 'delete'],
                   default='report',
                   help='report only (default); move duplicates to out-dir; '
                        'symlink survivors into out-dir/deduped; or delete '
                        'duplicates in place')
    p.add_argument('--review', action='store_true',
                   help='after processing, open an interactive window: the '
                        'group representative on the left, each other member '
                        'on the right one by one, with Keep / Delete buttons. '
                        'Deletes in place; overrides --action.')
    return p.parse_args()


def natural_key(path):
    m = re.findall(r'(\d+)', osp.basename(path))
    return (int(m[-1]) if m else 0, osp.basename(path))


def list_images(folder, recursive):
    pattern = '**/*' if recursive else '*'
    paths = glob(osp.join(folder, pattern), recursive=recursive)
    imgs = [p for p in paths
            if osp.isfile(p) and p.lower().endswith(IMG_EXTS)]
    return sorted(imgs, key=natural_key)


def is_dup(ssim, mse, psnr, args):
    """Return True if the (enabled) metrics mark this pair a duplicate."""
    checks = []
    if args.ssim is not None:
        checks.append(ssim >= args.ssim)
    if args.mse is not None:
        checks.append(mse <= args.mse)
    if args.psnr is not None:
        checks.append(psnr >= args.psnr)
    if not checks:
        raise SystemExit('All metrics disabled; enable at least one.')
    return all(checks) if args.require == 'all' else any(checks)


def parse_annotation(xml_path, img_hw=None):
    """Parse an XML into {class: [(x1,y1,x2,y2), ...]} in NORMALIZED [0,1]
    coordinates.

    Supports both COCO-style <bndbox><x><y><w><h> (as used elsewhere in this
    repo) and Pascal-VOC <bndbox><xmin><ymin><xmax><ymax>. Boxes are divided by
    the image width/height so differently-sized frames are comparable; the size
    is read from the XML <size> block, falling back to ``img_hw`` (h, w).
    """
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return None

    sz = root.find('size')
    w = h = None
    if sz is not None:
        w = float(sz.findtext('width') or 0) or None
        h = float(sz.findtext('height') or 0) or None
    if (not w or not h) and img_hw is not None:
        h, w = float(img_hw[0]), float(img_hw[1])
    if not w or not h:
        return None  # cannot normalize

    out = defaultdict(list)
    for o in root.findall('object'):
        name = (o.findtext('name') or '').strip()
        b = o.find('bndbox')
        if not name or b is None:
            continue
        if b.find('xmin') is not None:  # Pascal VOC
            x1 = float(b.findtext('xmin')); y1 = float(b.findtext('ymin'))
            x2 = float(b.findtext('xmax')); y2 = float(b.findtext('ymax'))
        elif b.find('x') is not None:  # COCO x y w h
            x = float(b.findtext('x')); y = float(b.findtext('y'))
            bw = float(b.findtext('w')); bh = float(b.findtext('h'))
            x1, y1, x2, y2 = x, y, x + bw, y + bh
        else:
            continue
        out[name].append((x1 / w, y1 / h, x2 / w, y2 / h))
    return out


def box_overlap(a, b, metric):
    """Overlap of two boxes: 'coverage' (inter / smaller) or 'iou'."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    if metric == 'iou':
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def annotation_verdict(ann_a, ann_b, args):
    """Decide whether annotations confirm two images are the same scene.

    Returns (verdict, detail):
      * 'unknown' -- one/both images have no usable annotation.
      * 'reject'  -- no shared class, or no shared-class box pair overlaps.
      * 'confirm' -- at least one shared class has an overlapping box pair.
    ``detail`` is '<class>:<overlap>' for the best confirming match, else ''.

    The images need only ONE class in common (annotations may miss others);
    that shared class must have boxes at an overlapping location, which is what
    separates a genuine duplicate from two different frames that happen to
    contain the same class.
    """
    if not ann_a or not ann_b:
        return 'unknown', ''
    shared = set(ann_a) & set(ann_b)
    if not shared:
        return 'reject', ''
    best_cls, best_ov = '', 0.0
    for cls in shared:
        for ba in ann_a[cls]:
            for bb in ann_b[cls]:
                ov = box_overlap(ba, bb, args.ann_overlap_metric)
                if ov > best_ov:
                    best_cls, best_ov = cls, ov
    if best_ov >= args.ann_overlap:
        return 'confirm', f'{best_cls}:{best_ov:.2f}'
    return 'reject', f'{best_cls}:{best_ov:.2f}'


def decide_pair(ssim, mse, psnr, ann_a, ann_b, args):
    """Combine the pixel metrics with the annotation gate.

    Returns (is_duplicate, verdict, detail) where verdict is the annotation
    outcome ('n/a' when --ann-dir is not used).
    """
    pixel_dup = is_dup(ssim, mse, psnr, args)
    if args.ann_dir is None:
        return pixel_dup, 'n/a', ''
    verdict, detail = annotation_verdict(ann_a, ann_b, args)
    if verdict == 'confirm':
        dup = pixel_dup
    elif verdict == 'reject':
        dup = False
    else:  # unknown: no annotation for one/both
        dup = pixel_dup if args.ann_missing == 'pixel-only' else False
    return dup, verdict, detail


class Union:
    """Minimal union-find over image indices."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def load_all(paths, size):
    """Load every image as a size x size float32 array in [0, 1].

    Returns (imgs, kept_paths, orig_hw) where orig_hw[k] is the (h, w) of the
    image before resizing (used to normalize annotation boxes).
    """
    imgs, kept, orig_hw = [], [], []
    for p in paths:
        g = _imread_gray(p)
        if g is None:
            print(f'[warn] unreadable, skipped: {p}')
            continue
        orig_hw.append(g.shape[:2])
        g = _resize(g, size).astype(np.float32) / 255.0
        imgs.append(g)
        kept.append(p)
    return imgs, kept, orig_hw


def find_annotation(ann_dir, img_path):
    """Locate the XML whose stem matches the image."""
    stem = osp.splitext(osp.basename(img_path))[0]
    cand = osp.join(ann_dir, stem + '.xml')
    return cand if osp.exists(cand) else None


def find_mask(mask_dir, img_path):
    """Locate the mask whose stem matches the image (any image extension)."""
    stem = osp.splitext(osp.basename(img_path))[0]
    for ext in IMG_EXTS:
        for cand in (osp.join(mask_dir, stem + ext),
                     osp.join(mask_dir, stem + ext.upper())):
            if osp.exists(cand):
                return cand
    return None


def load_annotations(ann_dir, paths, orig_hw):
    """For each image path, parse its annotation to {class: [norm boxes]}.

    Returns (anns, n_found) where anns[k] is None when no usable XML exists.
    """
    anns, n_found = [], 0
    for p, hw in zip(paths, orig_hw):
        xml = find_annotation(ann_dir, p)
        ann = parse_annotation(xml, hw) if xml else None
        if ann:
            n_found += 1
        anns.append(ann)
    return anns, n_found


def main():
    args = parse_args()
    out_dir = args.out_dir or osp.join(args.img_dir, '_dedup')

    paths = list_images(args.img_dir, args.recursive)
    if len(paths) < 2:
        raise SystemExit(f'Need >=2 images in {args.img_dir}, found {len(paths)}')

    print(f'Loading {len(paths)} images at {args.size}x{args.size}...')
    imgs, paths, orig_hw = load_all(paths, args.size)
    n = len(imgs)

    anns = [None] * n
    if args.ann_dir:
        anns, n_found = load_annotations(args.ann_dir, paths, orig_hw)
        print(f'Annotation gate ON: {n_found}/{n} images have usable XML; '
              f'overlap {args.ann_overlap_metric} >= {args.ann_overlap}; '
              f'missing -> {args.ann_missing}')

    enabled = [m for m, v in (('SSIM', args.ssim), ('MSE', args.mse),
                              ('PSNR', args.psnr)) if v is not None]
    print(f'Comparing {n} images pairwise ({n * (n - 1) // 2} pairs); '
          f'metrics: {", ".join(enabled)}; require={args.require}')

    uf = Union(n)
    pairs = []  # (i, j, ssim, mse, psnr, verdict, detail)
    rejected = []  # pixel-similar pairs the annotation gate vetoed
    data_range = 1.0
    win = min(7, args.size if args.size % 2 else args.size - 1)

    for i in range(n):
        a = imgs[i]
        for j in range(i + 1, n):
            b = imgs[j]
            mse = float(np.mean((a - b) ** 2))
            # Cheap MSE gate: clearly-different pairs skip the costlier SSIM.
            if (args.prefilter_mse is not None
                    and mse > args.prefilter_mse):
                continue
            psnr = (float('inf') if mse == 0
                    else sk_psnr(a, b, data_range=data_range))
            ssim = sk_ssim(a, b, data_range=data_range, win_size=win)
            dup, verdict, detail = decide_pair(
                ssim, mse, psnr, anns[i], anns[j], args)
            if dup:
                uf.union(i, j)
                pairs.append((i, j, ssim, mse, psnr, verdict, detail))
            elif args.ann_dir and verdict == 'reject' \
                    and is_dup(ssim, mse, psnr, args):
                # pixel metrics said duplicate; annotations overruled it
                rejected.append((i, j, ssim, mse, psnr, verdict, detail))
        if n > 50 and (i + 1) % 50 == 0:
            print(f'  ...{i + 1}/{n} rows compared')

    # Build groups (only those with >1 member are near-duplicate clusters).
    groups = {}
    for idx in range(n):
        groups.setdefault(uf.find(idx), []).append(idx)
    clusters = {root: sorted(mem) for root, mem in groups.items()
                if len(mem) > 1}

    removed = set()
    for mem in clusters.values():
        for idx in mem[1:]:  # keep earliest (mem[0]) as representative
            removed.add(idx)

    os.makedirs(out_dir, exist_ok=True)
    _write_reports(out_dir, paths, pairs, rejected, clusters, removed)

    print(f'\n{len(pairs)} near-duplicate pairs, {len(clusters)} groups, '
          f'{len(removed)} images flagged for removal '
          f'({n - len(removed)} survivors of {n}).')
    if args.ann_dir:
        print(f'{len(rejected)} pixel-similar pairs were vetoed by the '
              f'annotation gate (see rejected_by_annotation.csv).')
    print(f'Reports written under: {out_dir}')

    if args.review:
        print('\nOpening interactive review window '
              '(Keep / Delete per candidate)...')
        deleted = review_groups(
            paths, clusters, args.ann_dir, args.mask_dir)
        extra = [n for n, on in (('annotations', args.ann_dir),
                                 ('masks', args.mask_dir)) if on]
        note = (f' (matching {" & ".join(extra)} removed too)'
                if extra else '')
        print(f'Review done: deleted {len(deleted)} images in place{note}.')
        return

    _apply_action(args, out_dir, paths, removed)
    if args.action == 'report':
        print('Action=report: nothing moved/deleted. Review '
              f'{osp.join(out_dir, "remove_list.txt")} then rerun with '
              '--action move/delete or --review.')


def _write_pair_csv(path, paths, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_a', 'image_b', 'ssim', 'mse', 'psnr',
                    'ann_verdict', 'ann_match'])
        for i, j, ssim, mse, psnr, verdict, detail in rows:
            w.writerow([osp.basename(paths[i]), osp.basename(paths[j]),
                        f'{ssim:.4f}', f'{mse:.6f}', f'{psnr:.2f}',
                        verdict, detail])


def _write_reports(out_dir, paths, pairs, rejected, clusters, removed):
    _write_pair_csv(osp.join(out_dir, 'duplicates.csv'), paths, pairs)
    if rejected:
        _write_pair_csv(osp.join(out_dir, 'rejected_by_annotation.csv'),
                        paths, rejected)

    with open(osp.join(out_dir, 'groups.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'representative', 'duplicates'])
        for gid, (_, mem) in enumerate(sorted(clusters.items())):
            w.writerow([gid, osp.basename(paths[mem[0]]),
                        ';'.join(osp.basename(paths[k]) for k in mem[1:])])

    with open(osp.join(out_dir, 'remove_list.txt'), 'w') as f:
        for idx in sorted(removed):
            f.write(paths[idx] + '\n')


def _try_remove(path):
    try:
        os.remove(path)
        return True
    except OSError as e:
        print(f'[warn] could not delete {path}: {e}')
        return False


def delete_image(img_path, ann_dir, mask_dir=None):
    """Delete an image plus its matching XML (``ann_dir``) and mask
    (``mask_dir``) when those are set. Returns
    (image_deleted, annotation_deleted, mask_deleted)."""
    img_ok = osp.exists(img_path) and _try_remove(img_path)
    xml = find_annotation(ann_dir, img_path) if ann_dir else None
    ann_ok = bool(xml) and _try_remove(xml)
    mask = find_mask(mask_dir, img_path) if mask_dir else None
    mask_ok = bool(mask) and _try_remove(mask)
    return img_ok, ann_ok, mask_ok


def _extras_note(n_ann, n_mask):
    bits = []
    if n_ann:
        bits.append(f'{n_ann} annotations')
    if n_mask:
        bits.append(f'{n_mask} masks')
    return f' (+{" & ".join(bits)})' if bits else ''


def _apply_action(args, out_dir, paths, removed):
    if args.action == 'report' or not removed:
        return
    if args.action == 'move':
        img_dest = osp.join(out_dir, 'duplicates')
        ann_dest = osp.join(out_dir, 'duplicate_annotations')
        mask_dest = osp.join(out_dir, 'duplicate_masks')
        os.makedirs(img_dest, exist_ok=True)
        n_ann = n_mask = 0
        for idx in sorted(removed):
            p = paths[idx]
            shutil.move(p, osp.join(img_dest, osp.basename(p)))
            xml = find_annotation(args.ann_dir, p) if args.ann_dir else None
            if xml:
                os.makedirs(ann_dest, exist_ok=True)
                shutil.move(xml, osp.join(ann_dest, osp.basename(xml)))
                n_ann += 1
            mask = find_mask(args.mask_dir, p) if args.mask_dir else None
            if mask:
                os.makedirs(mask_dest, exist_ok=True)
                shutil.move(mask, osp.join(mask_dest, osp.basename(mask)))
                n_mask += 1
        print(f'Moved {len(removed)} duplicates to {img_dest}'
              + _extras_note(n_ann, n_mask))
    elif args.action == 'delete':
        n_ann = n_mask = 0
        for idx in sorted(removed):
            _, ann_ok, mask_ok = delete_image(
                paths[idx], args.ann_dir, args.mask_dir)
            n_ann += ann_ok
            n_mask += mask_ok
        print(f'Deleted {len(removed)} duplicates in place'
              + _extras_note(n_ann, n_mask))
    elif args.action == 'symlink':
        img_dest = osp.join(out_dir, 'deduped')
        ann_dest = osp.join(out_dir, 'deduped_annotations')
        mask_dest = osp.join(out_dir, 'deduped_masks')
        os.makedirs(img_dest, exist_ok=True)
        n_ann = n_mask = 0

        def _link(src, dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            link = osp.join(dest_dir, osp.basename(src))
            if not osp.lexists(link):
                os.symlink(osp.abspath(src), link)
                return True
            return False

        for idx, p in enumerate(paths):
            if idx in removed:
                continue
            _link(p, img_dest)
            xml = find_annotation(args.ann_dir, p) if args.ann_dir else None
            if xml:
                n_ann += _link(xml, ann_dest)
            mask = find_mask(args.mask_dir, p) if args.mask_dir else None
            if mask:
                n_mask += _link(mask, mask_dest)
        print(f'Symlinked {len(paths) - len(removed)} survivors into {img_dest}'
              + _extras_note(n_ann, n_mask))


def review_groups(paths, clusters, ann_dir=None, mask_dir=None):
    """Interactive keep/delete review, one group at a time.

    For each near-duplicate group the representative (always kept) is shown on
    the left and every other member is shown on the right in turn. "Keep" (or
    key ``k``) leaves the file; "Delete" (or key ``d``) removes it -- and its
    matching XML (``ann_dir``) and mask (``mask_dir``) when set -- from disk
    immediately. When a group
    is exhausted the next group starts automatically; closing the window stops
    early. Returns the list of deleted image paths.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
    except ImportError:
        raise SystemExit(
            '--review needs matplotlib: pip install matplotlib')
    if matplotlib.get_backend().lower() == 'agg':
        raise SystemExit(
            'matplotlib has no interactive display (Agg backend); run '
            '--review on a machine with a GUI, or use --action delete.')

    groups = [mem for _, mem in sorted(clusters.items())]
    if not groups:
        print('No near-duplicate groups to review.')
        return []

    deleted = []
    st = {'gi': 0, 'di': 1, 'done': False}  # di=1 skips the representative

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(bottom=0.18, wspace=0.05)
    b_keep = Button(fig.add_axes([0.28, 0.04, 0.18, 0.08]), 'Keep  (k)')
    b_del = Button(fig.add_axes([0.54, 0.04, 0.18, 0.08]), 'Delete  (d)')

    def _draw_img(ax, path, title):
        ax.clear()
        img = _imread_display(path) if osp.exists(path) else None
        if img is None:
            ax.text(0.5, 0.5, '(missing)', ha='center', va='center')
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    def show():
        gi, di = st['gi'], st['di']
        mem = groups[gi]
        rep, cur = paths[mem[0]], paths[mem[di]]
        _draw_img(axL, rep, f'KEEP (representative)\n{osp.basename(rep)}')
        _draw_img(axR, cur,
                  f'group {gi + 1}/{len(groups)}   '
                  f'candidate {di}/{len(mem) - 1}\n{osp.basename(cur)}')
        fig.suptitle(f'{len(deleted)} deleted so far — '
                     'Keep (k) / Delete (d), close window to stop',
                     fontsize=11)
        fig.canvas.draw_idle()

    def advance():
        st['di'] += 1
        while st['di'] >= len(groups[st['gi']]):
            st['gi'] += 1
            st['di'] = 1
            if st['gi'] >= len(groups):
                st['done'] = True
                plt.close(fig)
                return
        show()

    def on_keep(_event=None):
        if not st['done']:
            advance()

    def on_delete(_event=None):
        if st['done']:
            return
        cur = paths[groups[st['gi']][st['di']]]
        _, ann_ok, mask_ok = delete_image(cur, ann_dir, mask_dir)
        deleted.append(cur)
        extra = [n for n, ok in (('annotation', ann_ok), ('mask', mask_ok))
                 if ok]
        print(f'deleted: {cur}'
              + (f' (+{" & ".join(extra)})' if extra else ''))
        advance()

    def on_key(event):
        if event.key == 'k':
            on_keep()
        elif event.key == 'd':
            on_delete()

    b_keep.on_clicked(on_keep)
    b_del.on_clicked(on_delete)
    fig.canvas.mpl_connect('key_press_event', on_key)

    show()
    plt.show()
    return deleted


if __name__ == '__main__':
    main()
