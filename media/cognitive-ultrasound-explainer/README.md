# 认知超声：PPT 辅助动画 v2

55 秒 · 1920×1080 · 30 fps · H.264 / 8-bit 4:2:0 · 无音轨。纯白背景，扇形内保留黑底灰度合成图。由答辩者现场讲解，视频只含对象标签和一句取舍提示，没有片头、章节、页码、页脚、完整字幕或结尾口号。

## 预览与重新渲染

在本目录执行：
```powershell
npm ci
npm run dev
npm run stills
npm run render
python scripts/verify-video.py
```
Studio 选择 **CognitiveUltrasoundPPT**。输出全部写入被 Git 忽略的 ../../outputs/cognitive-ultrasound-explainer/ppt-v2；不会覆盖上一版 MP4。Windows 默认使用已安装的 Chrome，可用 REMOTION_BROWSER 指定路径。中文字体使用系统 Microsoft YaHei，其他平台请安装 Noto Sans CJK SC。校验脚本需要 Python、Pillow、numpy、opencv-python；Windows 使用 Remotion 自带 ffmpeg/ffprobe。

## 修改入口

- src/config.ts：颜色、四段时间、扇形位置、每帧 6 条示意线的方向及取舍演示周期。
- src/Composition.tsx：主扇形连续移动、逐线累积、持帧、两次换帧及估计显现。所有动作由帧号驱动。
- src/scenes/Pulse.tsx / Build.tsx：合并的发射、接收、逐线成像；Tradeoff.tsx：统一取舍对照；Reconstruct.tsx / Uncertainty.tsx：实测、估计、不确定性与下一帧。
- src/visuals.tsx：同一探头、扇形裁切、扫描线与热图。
- src/Root.tsx 与 scripts/render.mjs：合成注册、渲染参数与检查帧。改变时长时同步更新校验脚本。
- public/phantom：复用上一版固定种子 20260922 的合成纹理；不需要训练、科研推理或重新生成素材。

Captions.tsx、Closing.tsx 保留为空兼容组件。src/captions.json、docs/narration.md、docs/narration.srt 和 export-captions.mjs 是 **115 秒旧版档案**，不被 v2 导入，不要将其用于本版。需要现场讲解时使用 docs/presenter-notes.md。

## 示意假设与科学边界

全片是传统逐线聚焦扫描的简化原理示意，不代表所有超声采集方式。一条线的信息来自发射后接收的多个深度回波及后续处理；声束在发射时存在。最初声传播显著慢放，随后扫描与心脏更新采用另一教学时间尺度，均不代表设备真实速度。

取舍对照同时改变范围、深度和采样密度，是定性总览，不是控制变量实验。两侧使用同一运动时钟，较慢更新的一侧保持上一幅图更久；没有将心跳减速。深度和角度为画面坐标，24 / 9 条线、84 / 25 个视频帧的采集周期均是可观察动作的演示参数，不是实际 FPS、提速倍数或实验测量。图像信息更密不等于真实空间分辨率必然更高，实际还受焦点、声场和系统处理等因素影响。

稀疏观测与重建估计使用不同的合成素材，青蓝实线表示实测方向，紫色虚线标识估计。未采区域不保证正确。橙色表示重建不确定性，不是病灶、诊断概率或预知真实误差。热区及候选方向为解释用手工示意，没有运行 CASL 选线算法。

当前帧重建后，为下一帧选择扫描位置；两次换帧均为每帧 6 条线，重新计数，不在当前帧无限补采。第一帧分散方向只代表初始化示意。不确定性保留局部热区，不承诺每次测量全面改善。速度收益还依赖重建计算及设备实现，不宣称临床优势或设备实时运行。

全部图像是自行生成的简化四腔切面，不是患者影像或实验结果；未使用第三方音乐、论文图或官方视频。详细出处见 docs/sources.md。

## 交付与检查

- ppt-v2/cognitive-ultrasound-ppt-v2.mp4：新版成片。
- ppt-v2/contact-sheet.png：从最终 MP4 抽取的关键帧联系表。
- ppt-v2/half-size-*.png：960×540 缩小检查图。
- ppt-v2/final-previews、qa-results.json、ffprobe.json：编码后检查证据。
- docs/storyboard.md：四段分镜；docs/presenter-notes.md：现场讲解提纲；docs/qa.md：检查记录。

旧版 MP4 保留在父输出目录。源码沿用原项目，视频、node_modules、渲染缓存不提交；不交付源码压缩包。本次仅修改动画相关文件。
