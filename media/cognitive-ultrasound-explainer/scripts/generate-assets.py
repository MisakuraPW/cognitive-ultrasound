"""Fixed-seed educational phantom, NOT physical simulation or inference.
Speckle deforms coherently. Estimates are separately perturbed drawings.
Requires numpy, scipy, Pillow. Assets committed for npm-only reproduction.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image
OUT = Path(__file__).resolve().parents[1] / 'public' / 'phantom'
OUT.mkdir(parents=True, exist_ok=True)
N=48
W,H=720,560
y,x=np.mgrid[0:H,0:W].astype(float)
x=(x-W/2)/W*1000; y=y/H*600
rng=np.random.default_rng(20260922)
noise=gaussian_filter(rng.rayleigh(1,(H,W)),.6)
noise=noise/noise.mean()
def phantom(phase, variant=0):
    beat=(1-np.cos(phase*2*np.pi))/2
    xx=x/(1-.025*beat); yy=y+7*beat
    texture=map_coordinates(noise,[(yy/600*H)%H,((xx/1000+.5)*W)%W],order=1,mode='reflect')
    tissue=np.full((H,W),22.)
    for cx,cy,rx,ry,ang in [(-78,270,83,141,-.22),(90,266,68,126,.24),(-100,466,92,74,.08),(104,457,81,69,-.12)]:
        cx+=variant*(5 if cx>0 else -3); cy+=variant*(3 if cy<350 else -3)
        c,s=np.cos(ang),np.sin(ang); dx=xx-cx;dy=yy-cy
        u=dx*c+dy*s;v=-dx*s+dy*c
        contraction=1-(.17 if cy<350 else -.04)*beat
        d=np.sqrt((u/(rx*contraction+variant*2))**2+(v/(ry*(1-.06*beat)))**2)
        theta=np.arctan2(v/(ry*(1-.06*beat)),u/(rx*contraction+variant*2))
        d=d*(1+.045*np.sin(3*theta+cx/90)+.025*np.cos(5*theta+phase*2*np.pi))
        cavity=1/(1+np.exp(np.clip((d-.88)*35,-50,50)))
        tissue=tissue*(1-.82*cavity)
        tissue+=125*np.exp(-((d-1)/.115)**2)+34*np.exp(-((d-1.19)/.19)**2)
    for cx in [-84,92]:
        valve_y=383+13*np.sin(phase*2*np.pi)+(xx-cx)*.14
        tissue+=55*np.exp(-((yy-valve_y)/4.2)**2)*np.exp(-((xx-cx)/58)**6)
    tissue+=17*np.exp(-((np.sqrt((xx/255)**2+((yy-350)/245)**2)-1)/.09)**2)
    val=np.clip(tissue*(.35+.65*texture)+3*texture,0,240)
    val*=.84+.16*np.exp(-((yy-290)/250)**2)
    return Image.fromarray(np.uint8(val),'L')
for i in range(N):
    phantom(i/N).save(OUT/f'observed-{i:02}.png',optimize=True)
    phantom(i/N,1).save(OUT/f'estimate-{i:02}.png',optimize=True)
for v in [-1,0,1]:phantom(0,v*2+1).save(OUT/f'possible-{v+1}.png',optimize=True)
print(f'Wrote {2*N+3} synthetic images')
