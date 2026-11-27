"""vc-run npy-selftest against numpy itself: RNG draws and linalg bit for bit.
usage: python tests/npy_selftest.py [vc-run.exe]"""
import json, subprocess, sys
import numpy as np
exe = sys.argv[1] if len(sys.argv) > 1 else r'build\vc-run.exe'
vc = json.loads(subprocess.run([exe, 'npy-selftest'], capture_output=True, text=True).stdout)
g = np.random.default_rng(0)
seq = []
for i in range(60):
    pop, size = 3 + (i * 7) % 23, 2 + i % 2
    seq.append([int(v) for v in g.choice(pop, size=size, replace=False)])
    seq.append(int(g.integers(1 + i % 4)))
P = np.float64([1.5, 2.25, 3.1, 4.7, 5.2, 6.9, 7.3, 8.8, 9.1, 10.4, 11.6, 12.2]).reshape(6, 2)
A = np.column_stack([P[:, 0], P[:, 1], np.ones(6)]); b = P[:, 0] ** 2 + P[:, 1] ** 2
x = np.float64([-0.6, 0.8]); Pf = P.astype(np.float32) * np.float32(1.37)
py = {'rng': seq, 'svd': np.linalg.svd(P)[2].ravel().tolist(), 'lstsq': np.linalg.lstsq(A, b, rcond=None)[0].tolist(),
      'matvec': (P @ x).tolist(), 'matvec_f32': (Pf @ np.float32([-0.6, 0.8])).astype(float).tolist(), 'dot': float(P[0] @ P[1])}
print('blas:', vc['blas'])
bad = 0
for k, v in py.items():
    same = v == vc[k]
    bad += not same
    print('%-10s %s' % (k, 'identical' if same else 'DIFFERENT\n  numpy %s\n  vc    %s' % (str(v)[:300], str(vc[k])[:300])))
sys.exit(1 if bad else 0)
