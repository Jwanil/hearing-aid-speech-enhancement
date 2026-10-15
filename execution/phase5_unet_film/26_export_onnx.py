# execution/phase5_unet_film/26_export_onnx.py
import argparse, os
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(f"[export] stub ONNX: {args.ckpt} -> {args.out}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("# onnx stub\n")
if __name__ == "__main__":
    main()
