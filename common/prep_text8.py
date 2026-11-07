"""text8 (Mahoney): 100M chars, 27 simbolos (a-z + espacio). Particion estandar 90M/5M/5M:
train = primeros 90M; val = ULTIMOS 5M (el 'test' estandar); los 5M intermedios (valid estandar) no se usan."""
import numpy as np, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = open(os.path.join(ROOT, "data", "raw_text8", "text8"), "rb").read()
assert len(raw) == 100_000_000
arr = np.frombuffer(raw, dtype=np.uint8)
itos = [" "] + [chr(c) for c in range(97, 123)]
lut = np.zeros(256, dtype=np.uint8)
for i, ch in enumerate(itos):
    lut[ord(ch)] = i
ids = lut[arr]
out = os.path.join(ROOT, "data", "text8"); os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, "train.npy"), ids[:90_000_000])
np.save(os.path.join(out, "val.npy"), ids[95_000_000:])
json.dump(itos, open(os.path.join(out, "vocab.json"), "w"))
print("train", 90_000_000, "val", 5_000_000, "V", len(itos))
