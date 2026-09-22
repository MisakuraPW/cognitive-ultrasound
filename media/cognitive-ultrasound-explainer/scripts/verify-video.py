"""Verify PPT v2 encoding and temporal behavior; never modify v1 outputs."""
from pathlib import Path
import subprocess, json, hashlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np
root=Path(__file__).resolve().parents[1]
parent=(root/'../../outputs/cognitive-ultrasound-explainer').resolve()
out=parent/'ppt-v2'; video=out/'cognitive-ultrasound-ppt-v2.mp4'
bins=root/'node_modules/@remotion/compositor-win32-x64-msvc'
ffmpeg=str(bins/'ffmpeg.exe'); ffprobe=str(bins/'ffprobe.exe')
probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams','-show_format','-of','json',str(video)]))
(out/'ffprobe.json').write_text(json.dumps(probe,indent=2),encoding='utf-8')
v=next(s for s in probe['streams'] if s['codec_type']=='video')
assert (v['width'],v['height'])==(1920,1080)
assert v['codec_name']=='h264' and v['pix_fmt'] in ('yuv420p','yuvj420p')
assert v['r_frame_rate']=='30/1' and int(v['nb_frames'])==1650
assert abs(float(probe['format']['duration'])-55)<.01
assert not any(s['codec_type']=='audio' for s in probe['streams'])
subprocess.run([ffmpeg,'-v','error','-i',str(video),'-c:v','rawvideo','-f','null','-'],check=True)
# Decode every frame at quarter size to inspect temporal continuity, not only stills.
import cv2
cap=cv2.VideoCapture(str(video))
small=[]
while True:
 ok, frame=cap.read()
 if not ok: break
 small.append(cv2.cvtColor(cv2.resize(frame,(480,270)),cv2.COLOR_BGR2GRAY))
cap.release()
assert len(small)==1650
seq=np.stack(small)
change=np.abs(np.diff(seq.astype(np.int16),axis=0)).mean(axis=(1,2))
scan_motion=float(change[150:285].mean())
ending_motion=float(change[1590:].max())
assert scan_motion>.02,('Scan sequence must move',scan_motion)
assert ending_motion<.5,('Final 2 seconds must be held',ending_motion)
np.savetxt(out/'frame-motion.csv',np.column_stack([np.arange(1,1650)/30,change]),delimiter=',',header='time_seconds,mean_absolute_change',comments='')
p=out/'final-previews'; p.mkdir(exist_ok=True)
keys=[36,210,315,420,540,810,990,1155,1215,1281,1419,1620]
frames=set(keys+[0,537,543,1590,1649])
# Sequence checks surround scanning, layout transitions and both frame changes.
for start,end,step in [(0,150,15),(150,285,15),(351,399,6),(708,771,9),(1200,1365,15),(1386,1575,15)]:
 frames.update(range(start,end+1,step))
for f in sorted(frames):
 subprocess.run([ffmpeg,'-v','error','-y','-ss',str(f/30),'-i',str(video),'-frames:v','1',str(p/f'{f:04d}.png')],check=True)
def im(f): return np.asarray(Image.open(p/f'{f:04d}.png').convert('RGB')).astype(float)
a=im(537); b=im(543); diff=np.abs(a-b)
left=float(diff[430:790,200:800].mean())
right=float(diff[430:680,1190:1620].mean())
assert left<1.5,('Long acquisition must hold',left)
assert right>left+.5,('Short acquisition should update',right,left)
held=float(np.abs(im(1590)-im(1649)).mean())
assert held<.5,('Last two seconds should hold',held)
white_min=min(float(im(f)[:30,:30].min()) for f in keys)
assert white_min>=250,('White background',white_min)
old_hash=hashlib.sha256((parent/'cognitive-ultrasound-explainer.mp4').read_bytes()).hexdigest().upper()
assert old_hash=='C47165A8B2CB4CD97B5036AFC7E2B9ECE5A52293CC378C902FF0F4E3D3F53BA6'
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
def sheet(fs,name):
 cols=3; rows=(len(fs)+cols-1)//cols
 canvas=Image.new('RGB',(1920,rows*392),'white'); d=ImageDraw.Draw(canvas)
 for i,f in enumerate(fs):
  x=(i%cols)*640; y=(i//cols)*392
  canvas.paste(Image.open(p/f'{f:04d}.png').resize((640,360)),(x,y))
  d.text((x+18,y+364),f'{f/30:05.2f} s',font=font,fill='#4b5963')
 canvas.save(out/name)
sheet(keys,'contact-sheet.png')
sheet([1200,1215,1230,1245,1260,1275,1290,1305,1320,1335,1350,1365],'transition-first.png')
sheet([1386,1401,1416,1431,1446,1461,1476,1491,1506,1521,1536,1551],'transition-second.png')
sheet([0,15,30,45,60,75,90,105,120,150,195,255],'scan-sequence.png')
for f,name in [(540,'tradeoff'),(1215,'uncertainty'),(1620,'ending')]:
 Image.open(p/f'{f:04d}.png').resize((960,540)).save(out/f'half-size-{name}.png')
report={'duration_seconds':55,'frames':1650,'fps':30,'width':1920,'height':1080,
 'codec':'h264','pixel_format':v['pix_fmt'],'audio':'none','decode':'passed',
 'all_frames_temporally_checked':1650,'scan_motion_MAE':scan_motion,'ending_max_frame_MAE':ending_motion,'extracted_frames':len(frames),'slow_side_hold_MAE':left,'fast_side_update_MAE':right,
 'ending_hold_MAE':held,'white_corner_min':white_min,'old_mp4_sha256_unchanged':old_hash}
(out/'qa-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
