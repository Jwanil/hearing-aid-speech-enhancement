# execution/phase5_unet_film/25_export_tflite.py
# Export best_composite.pt -> TFLite FP16 (phone) or INT8 (ESP32) per Directive 05 §8.1
# Requires on pod: pip install onnx onnx-tf or ai-edge-litert
import argparse, os, torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--quant", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(f"[export] stub: {args.ckpt} --{args.quant} -> {args.out}")
    print("Full TFLite export: implement via torch.onnx.export -> onnx2tf or Litert converter in pod.")
    # Skeleton: load model, trace, export ONNX as intermediate
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("# tflite stub — replace with converter output\n")

if __name__ == "__main__":
    main()
