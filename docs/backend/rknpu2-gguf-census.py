# Minimal GGUF header parser: per-tensor name/dims/type, then compute the
# W4A4 K-padding inflation (Hadamard pads K to next_power_of_two).
import struct, sys
f = open(sys.argv[1], 'rb')
def u32(): return struct.unpack('<I', f.read(4))[0]
def u64(): return struct.unpack('<Q', f.read(8))[0]
def s():   return f.read(u64()).decode('utf-8', 'replace')
def val(t):
    if t == 0: return struct.unpack('<B', f.read(1))[0]
    if t == 1: return struct.unpack('<b', f.read(1))[0]
    if t == 2: return struct.unpack('<H', f.read(2))[0]
    if t == 3: return struct.unpack('<h', f.read(2))[0]
    if t == 4: return u32()
    if t == 5: return struct.unpack('<i', f.read(4))[0]
    if t == 6: return struct.unpack('<f', f.read(4))[0]
    if t == 7: return struct.unpack('<B', f.read(1))[0]
    if t == 8: return s()
    if t == 9:
        et = u32(); n = u64()
        return [val(et) for _ in range(n)]
    if t == 10: return u64()
    if t == 11: return struct.unpack('<q', f.read(8))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]
    raise Exception(f'kv type {t}')
assert f.read(4) == b'GGUF'
ver = u32(); nt = u64(); nkv = u64()
for _ in range(nkv):
    k = s(); t = u32(); val(t)
def np2(x):
    p = 1
    while p < x: p *= 2
    return p
Q40 = 2  # ggml type ids
rows = []
for _ in range(nt):
    name = s(); nd = u32()
    dims = [u64() for _ in range(nd)]
    typ = u32(); u64()
    rows.append((name, dims, typ))
nominal = padded = 0
by_k = {}
for name, dims, typ in rows:
    if typ != Q40 or len(dims) != 2: continue
    K, N = dims[0], dims[1]
    nb = K * N // 2            # int4 packed at true K
    pb = np2(K) * N // 2       # packed at Hadamard-padded K_op
    nominal += nb; padded += pb
    key = (K, np2(K))
    e = by_k.setdefault(key, [0, 0, 0])
    e[0] += 1; e[1] += nb; e[2] += pb
print(f'Q4_0 2D tensors -> W4A4: nominal int4 {nominal/1e9:.2f} GB, K-padded {padded/1e9:.2f} GB, inflation {padded/nominal:.2f}x')
for (K, K_op), (cnt, nb, pb) in sorted(by_k.items(), key=lambda x: -x[1][2]):
    print(f'  K={K:>6} -> K_op={K_op:>6}  x{cnt:<4} nominal {nb/1e9:.2f} GB  padded {pb/1e9:.2f} GB  ({K_op/K:.2f}x)')
