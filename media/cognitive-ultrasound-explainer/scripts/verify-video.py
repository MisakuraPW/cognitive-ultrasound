"""Inspect encoded deliverable, generate contact sheet, verify hold-vs-update."""
from pathlib import Path
import subprocess,json
from PIL import Image,ImageDraw,ImageFont
import numpy as np
root=Path(__file__).resolve().parents[1]
out=root/'../../outputs/cognitive-ultrasound-explainer'
out=out.resolve(); video=out/'cognitive-ultrasound-explainer.mp4'
b=root/'node_modules/@remotion/compositor-win32-x64-msvc'
ffmpeg=str(b/'ffmpeg.exe'); ffprobe=str(b/'ffprobe.exe')
probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams','-show_format','-of','json',str(video)]))
(out/'ffprobe.json').write_text(json.dumps(probe,ensure_ascii=False,indent=2),encoding='utf-8')
v=next(s for s in probe['streams'] if s['codec_type']=='video')
assert (v['width'],v['height'])==(1920,1080)
assert v['r_frame_rate']=='30/1'
assert int(v['nb_frames'])==3450
assert abs(float(probe['format']['duration'])-115)<.01
assert not any(s['codec_type']=='audio' for s in probe['streams'])
subprocess.run([ffmpeg,'-v','error','-i',str(video),'-c:v','rawvideo','-f','null','-'],check=True)
p=out/'final-previews';p.mkdir(exist_ok=True)
frames=set([0,210,270,360,540,720,840,990,1200,1320,1470,1485,1620,1770,1950,2100,2280,2490,2640,2820,2910,2916,2940,3030,3036,3060,3240,3390,3449])
for f in [660,1140,1860,2400,3150]:
 frames.update([f-1,f,f+15])
for f in sorted(frames):
 subprocess.run([ffmpeg,'-v','error','-y','-ss',str(f/30),'-i',str(video),'-frames:v','1',str(p/f'{f:04d}.png')],check=True)
a=np.asarray(Image.open(p/'1470.png')).astype(float)
b=np.asarray(Image.open(p/'1485.png')).astype(float)
diff=np.abs(a-b)
left=float(diff[355:725,220:790].mean())
right=float(diff[355:725,1120:1690].mean())
assert left<1.5,('Dense side should hold previous image',left)
assert right>left+1,('Sparse side should update earlier',right,left)
keys=[210,540,840,990,1320,1470,1770,2100,2280,2640,2940,3390]
canvas=Image.new('RGB',(1920,4*398),'#14232d')
draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
for i,f in enumerate(keys):
 tile=Image.open(p/f'{f:04d}.png').convert('RGB').resize((640,360))
 x=(i%3)*640;y=(i//3)*398;canvas.paste(tile,(x,y))
 draw.text((x+16,y+365),f'{f/30:05.1f} s  /  合成原理示意',font=font,fill='#dce9eb')
canvas.save(out/'contact-sheet.png')
report={'duration_seconds':115,'frames':3450,'fps':30,'width':1920,'height':1080,'audio':'none (intentional)','decode':'passed','extracted_frames':len(frames),'dense_hold_MAE':left,'sparse_update_MAE':right,'subtitle_cues':19}
(out/'qa-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
