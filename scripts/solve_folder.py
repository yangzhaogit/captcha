
import argparse, os
from pathlib import Path
from captcha_solver import Captcha

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--dst_dir", required=True)
    ap.add_argument("--q", type=float, default=0.10)
    args = ap.parse_args()

    Path(args.dst_dir).mkdir(parents=True, exist_ok=True)
    cap = Captcha(model_path=args.model, q=args.q)

    imgs = sorted(Path(args.src_dir).glob("*.jpg"))
    for p in imgs:
        name = p.stem  # e.g., input21
        out = Path(args.dst_dir) / f"{name}.txt"
        cap(str(p), str(out))
        print(f"{name} -> {out.read_text().strip()}")

if __name__ == "__main__":
    main()
