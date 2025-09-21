
import os
import glob
import json
import math
import argparse
import pathlib
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _background_color(image: Image.Image) -> np.ndarray:
    """Estimate background color from image borders (robust median over border pixels)."""
    arr = np.asarray(image).astype(np.int16)
    H, W, C = arr.shape
    border = np.concatenate([
        arr[0, :, :],
        arr[-1, :, :],
        arr[:, 0, :],
        arr[:, -1, :],
    ], axis=0)
    bg = np.median(border, axis=0)
    return bg


def _binarize_by_bg_distance(image: Image.Image) -> np.ndarray:
    """Binarize by thresholding distance from the estimated background color using Otsu."""
    arr = np.asarray(image).astype(np.int16)
    bg = _background_color(image)
    dist = np.sqrt(np.sum((arr - bg) ** 2, axis=2))
    if dist.max() == 0:
        scaled = dist.astype(np.uint8)
    else:
        scaled = (255.0 * (dist / dist.max())).astype(np.uint8)

    # Otsu threshold
    hist, _ = np.histogram(scaled.flatten(), bins=256, range=(0, 255))
    total = scaled.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    varMax = 0.0
    threshold = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        varBetween = wB * wF * (mB - mF) ** 2
        if varBetween > varMax:
            varMax = varBetween
            threshold = i

    binary = scaled >= threshold
    # Ensure text appears as True (foreground smaller than background)
    if binary.mean() > 0.5:
        binary = ~binary
    return binary


def _connected_components(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return bounding boxes (top,left,bottom,right) for 4-connected components above a small area threshold."""
    H, W = binary.shape
    labels = -np.ones((H, W), dtype=np.int32)
    comps: List[Tuple[int,int,int,int]] = []
    label = 0
    for i in range(H):
        for j in range(W):
            if binary[i, j] and labels[i, j] == -1:
                # BFS
                q = [(i, j)]
                labels[i, j] = label
                min_i, max_i = i, i
                min_j, max_j = j, j
                idx = 0
                area = 0
                while idx < len(q):
                    x, y = q[idx]; idx += 1
                    area += 1
                    for nx, ny in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
                        if 0 <= nx < H and 0 <= ny < W and binary[nx, ny] and labels[nx, ny] == -1:
                            labels[nx, ny] = label
                            q.append((nx, ny))
                            if nx < min_i: min_i = nx
                            if nx > max_i: max_i = nx
                            if ny < min_j: min_j = ny
                            if ny > max_j: max_j = ny
                # Area filter to drop tiny specks
                if area >= 8:
                    comps.append((min_i, min_j, max_i+1, max_j+1))
                    label += 1
                else:
                    labels[labels == label] = -1
    return comps


def _crop_and_normalize(binary: np.ndarray, bbox: Tuple[int,int,int,int], out_size=(16,16)) -> np.ndarray:
    top, left, bottom, right = bbox
    crop = binary[top:bottom, left:right]
    h, w = crop.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=bool)
    y0 = (size - h)//2
    x0 = (size - w)//2
    padded[y0:y0+h, x0:x0+w] = crop
    im = Image.fromarray((padded*255).astype(np.uint8))
    im = im.resize(out_size, resample=Image.Resampling.NEAREST)
    arr = np.array(im) > 127
    return arr


def _segment_characters(image: Image.Image) -> List[np.ndarray]:
    binary = _binarize_by_bg_distance(image)
    comps = _connected_components(binary)
    boxes = sorted(comps, key=lambda b: b[1])
    if len(boxes) > 5:
        # merge nearest neighbor boxes until exactly 5
        while len(boxes) > 5:
            gaps = [boxes[i+1][1] - boxes[i][3] for i in range(len(boxes)-1)]
            k = int(np.argmin(gaps))
            t1,l1,b1,r1 = boxes[k]; t2,l2,b2,r2 = boxes[k+1]
            nt, nl, nb, nr = min(t1,t2), min(l1,l2), max(b1,b2), max(r1,r2)
            boxes = boxes[:k] + [(nt,nl,nb,nr)] + boxes[k+2:]
    elif len(boxes) < 5:
        H, W = binary.shape
        slice_w = W // 5
        new_boxes = []
        for i in range(5):
            x0 = i*slice_w
            x1 = W if i==4 else (i+1)*slice_w
            sub = binary[:, x0:x1]
            comps = _connected_components(sub)
            if comps:
                areas = [ (b[2]-b[0])*(b[3]-b[1]) for b in comps ]
                idx = int(np.argmax(areas))
                t,l,b,r = comps[idx]
                new_boxes.append((t, l+x0, b, r+x0))
            else:
                new_boxes.append((0, x0, H, x1))
        boxes = new_boxes

    chars = [_crop_and_normalize(binary, b) for b in boxes[:5]]
    return chars


def _char_templates_from_dataset(dataset_root: str) -> Dict[str, List[np.ndarray]]:
    """Build a dict: char -> list of 16x16 binary arrays from labeled images.
       Expects dataset_root to contain 'input/inputNN.jpg' and 'output/outputNN.txt' pairs.
    """
    in_dir = os.path.join(dataset_root, "input")
    out_dir = os.path.join(dataset_root, "output")
    image_paths = sorted(glob.glob(os.path.join(in_dir, "input*.jpg")))
    templates: Dict[str, List[np.ndarray]] = {}
    for img_path in image_paths:
        stem = pathlib.Path(img_path).stem  # inputXX
        idnum = stem.replace("input", "")
        gt_path = os.path.join(out_dir, f"output{idnum}.txt")
        if not os.path.exists(gt_path):
            continue  # skip unlabeled
        gt = open(gt_path, "r").read().strip()
        im = _load_image(img_path)
        char_imgs = _segment_characters(im)
        for ch_img, label_ch in zip(char_imgs, gt):
            templates.setdefault(label_ch, []).append(ch_img.astype(np.uint8))
    return templates


def _build_prototypes(templates: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    prototypes: Dict[str, np.ndarray] = {}
    for ch, imgs in templates.items():
        if len(imgs) == 1:
            avg = imgs[0].astype(np.float32)
        else:
            avg = np.mean([img.astype(np.float32) for img in imgs], axis=0)
        mx = float(avg.max()) if float(avg.max()) > 0 else 1.0
        prototypes[ch] = (avg / mx).astype(np.float32)
    return prototypes


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = a.flatten().astype(np.float32)
    vb = b.flatten().astype(np.float32)
    va = (va > 0.5).astype(np.float32)  # binarize input char
    denom = (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-8)
    return float(np.dot(va, vb) / denom)


def _predict_string(image: Image.Image, prototypes: Dict[str, np.ndarray]) -> str:
    char_imgs = _segment_characters(image)
    keys = list(prototypes.keys())
    protos = [prototypes[k] for k in keys]
    out_chars: List[str] = []
    for ci in char_imgs:
        scores = [_cosine_similarity(ci.astype(np.float32), p) for p in protos]
        idx = int(np.argmax(scores))
        out_chars.append(keys[idx])
    return "".join(out_chars)


def _save_model(prototypes: Dict[str, np.ndarray], path: str) -> None:
    keys = sorted(prototypes.keys())
    arr = np.stack([prototypes[k] for k in keys], axis=0)
    np.savez_compressed(path, keys=np.array(keys), prototypes=arr)


def _load_model(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    keys = [k for k in data["keys"]]
    arr = data["prototypes"]
    return { str(keys[i]): arr[i] for i in range(len(keys)) }


class Captcha(object):
    def __init__(self, model_path: str = "captcha_model.npz", dataset_root: str = None):
        """If model_path exists, load it. Otherwise, if dataset_root is provided, train and save. Else raise."""
        self.model_path = model_path
        if os.path.exists(model_path):
            self.prototypes = _load_model(model_path)
        elif dataset_root is not None:
            templates = _char_templates_from_dataset(dataset_root)
            if not templates:
                raise RuntimeError("No templates built — check dataset_root structure.")
            self.prototypes = _build_prototypes(templates)
            _save_model(self.prototypes, model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path} and no dataset_root provided to train.")

    def __call__(self, im_path: str, save_path: str):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        image = _load_image(im_path)
        text = _predict_string(image, self.prototypes)
        with open(save_path, "w") as f:
            f.write(text + "\n")
        return text


def main():
    parser = argparse.ArgumentParser(description="Simple template-matching CAPTCHA solver for fixed-style 5-char images.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train prototypes from dataset and save a model.")
    p_train.add_argument("--data", required=True, help="Path to dataset root containing input/ and output/ folders.")
    p_train.add_argument("--model", default="captcha_model.npz", help="Output model path.")

    p_infer = sub.add_parser("infer", help="Run inference on a single image.")
    p_infer.add_argument("--image", required=True, help="Path to input image (.jpg).")
    p_infer.add_argument("--model", default="captcha_model.npz", help="Model path to load.")
    p_infer.add_argument("--out", required=True, help="Path to output text file to write prediction.")

    args = parser.parse_args()

    if args.cmd == "train":
        templates = _char_templates_from_dataset(args.data)
        prototypes = _build_prototypes(templates)
        _save_model(prototypes, args.model)
        print(f"Saved model with {len(prototypes)} prototypes to {args.model}")
    elif args.cmd == "infer":
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        prototypes = _load_model(args.model)
        image = _load_image(args.image)
        pred = _predict_string(image, prototypes)
        with open(args.out, "w") as f:
            f.write(pred + "\n")
        print(pred)


if __name__ == "__main__":
    main()
