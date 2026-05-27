import numpy as np
import glob
import os


def inspect_npz(path):
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return
    print(f"File: {path}")
    keys = list(npz.keys())
    print(" Keys:", keys)
    for k in keys:
        v = npz[k]
        try:
            shape = getattr(v, "shape", None)
            dtype = getattr(v, "dtype", None)
            size = getattr(v, "size", None)
            print(f"  - {k}: shape={shape} dtype={dtype} size={size}")
            if size is not None and size > 0 and v.size <= 20:
                print(f"    sample=", v.tolist())
            elif size is not None and size > 0:
                flat = np.asarray(v).ravel()
                print(f"    sample(flat first 5)=", flat[:5].tolist())
        except Exception as e:
            print(f"    (error reading key {k}: {e})")
    print()


def main():
    files = sorted(glob.glob(os.path.join("data", "processed", "*.npz")))
    if not files:
        print("No npz files found in data/processed")
        return
    for f in files:
        inspect_npz(f)


if __name__ == '__main__':
    main()
