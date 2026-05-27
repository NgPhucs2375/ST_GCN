import numpy as np
from pathlib import Path
from collections import Counter
import json
import time

SRC = Path("data/processed/train_data.npz")
OUT4 = Path("data/processed/train_data_4ch.npz")
OUT6 = Path("data/processed/train_data_6ch.npz")


def make_meta(labels, channels):
    cnt = Counter(labels.tolist())
    return {
        "source": str(SRC),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).relative_to(Path.cwd())),
        "channels": channels,
        "num_samples": int(len(labels)),
        "class_counts": dict(cnt),
    }


def main():
    if not SRC.exists():
        print("Source not found:", SRC)
        return
    data = np.load(SRC, allow_pickle=True)
    seq = data['sequences']  # (N, T, V, C)
    labels = data['labels']
    print('loaded', SRC, 'shape', seq.shape)

    # Assume channel order: [x,y,z, vx,vy,vz, ax,ay,az]
    # 4ch: x,y,vx,vy -> indices [0,1,3,4]
    idx4 = [0,1,3,4]
    seq4 = seq[..., idx4]
    meta4 = make_meta(labels, ['x','y','vx','vy'])
    np.savez_compressed(OUT4, sequences=seq4, labels=labels, meta=np.array([json.dumps(meta4)], dtype=object))
    print('wrote', OUT4, 'shape', seq4.shape)

    # 6ch: x,y,z,vx,vy,vz -> indices [0,1,2,3,4,5]
    idx6 = [0,1,2,3,4,5]
    seq6 = seq[..., idx6]
    meta6 = make_meta(labels, ['x','y','z','vx','vy','vz'])
    np.savez_compressed(OUT6, sequences=seq6, labels=labels, meta=np.array([json.dumps(meta6)], dtype=object))
    print('wrote', OUT6, 'shape', seq6.shape)


if __name__ == '__main__':
    main()
