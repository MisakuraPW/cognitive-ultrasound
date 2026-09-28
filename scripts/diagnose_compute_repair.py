"""Bounded GPU diagnosis; writes evidence outside the immutable v2 batch."""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from cognitive_ultrasound.torch_casl.native import CapturedFrame, FrozenGraph, NativeCASL, nchw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = NativeCASL(FrozenGraph(root / 'export').to('cuda')).to('cuda')
    model.step_fn = torch.compile(model.dps_step, fullgraph=True)
    model.select = torch.compile(model.select, fullgraph=True)
    ref = next((root / '.cache/debug/b14/42').glob('*.h5'))
    with h5py.File(ref) as h:
        previous = h['0/resume_posterior_samples'][()]
        mask = h['0/resume_mask'][()][None]
        measurement = h['1/resume_buffer'][()][None]
        noise = h['1/noise'][()]
    tensors = (nchw(measurement, 'cuda').expand(2, -1, -1, -1), nchw(mask, 'cuda'), nchw(previous, 'cuda'), nchw(noise, 'cuda'))
    probe_file = out / 'lock_probe.h5'
    handle = h5py.File(probe_file, 'w')
    handle['x'] = [1]
    model.frame(*tensors, steps=50)
    handle.close()
    holders = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (proc / 'fd').iterdir():
                if os.readlink(fd) == str(probe_file):
                    holders.append({'pid': proc.name, 'fd': fd.name, 'cmd': (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')})
        except (OSError, PermissionError):
            pass
    try:
        with h5py.File(probe_file) as h:
            locked = False
    except BlockingIOError:
        locked = True
    results = {'lock_reproduced': locked, 'holders': holders, 'graphs': []}
    print(json.dumps(results), flush=True)
    for steps in (1, 2, 50):
        graph = CapturedFrame(model, tensors, steps)
        for factor in (1., .97, .83, 1.):
            changed = (tensors[0]*factor, tensors[1], tensors[2]*factor, tensors[3]*factor)
            actual = graph(*changed)
            expected = model.frame(*changed, steps=steps)
            torch.cuda.synchronize()
            diffs = [float((a.float()-b.float()).abs().max()) for a,b in zip(actual,expected)]
            results['graphs'].append(dict(steps=steps, factor=factor, max_abs=diffs))
            print(json.dumps(results['graphs'][-1]), flush=True)
    (out / 'diagnosis.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
