# 认知超声中文科普动画
115 秒 · 1920×1080 · 30 fps · Remotion 4.0.526。完整可编辑，固定随机种子，全部为合成原理示意。当前版本为带烧录字幕的静音 MP4，没有配音或音乐。

## 预览
在本目录执行：
```powershell
npm ci
npm run dev
```
打开命令实际返回的本地 Studio 地址，选择 CognitiveUltrasound。中文字体使用系统 Microsoft YaHei，其他平台可安装 Noto Sans CJK SC。

## 渲染
```powershell
npm run captions
npm run stills
npm run render
```
输出写入仓库被忽略的 outputs/cognitive-ultrasound-explainer。Windows 默认使用已安装的 Chrome；可通过 REMOTION_BROWSER 指定浏览器路径。其他系统交由 Remotion 查找/下载渲染浏览器。本项目的程序化渲染参数位于 scripts/render.mjs。

## 修改
- src/config.ts：颜色、分镜时刻、示意线数与位置。
- src/scenes/：六幕的文字及画面；src/Composition.tsx 负责同一探头/扇形跨幕衔接、时钟与持帧逻辑。
- src/captions.json：完整旁白及毫秒起止时间（Caption 格式）。修改后运行 npm run captions 同步 SRT 和旁白稿。
- src/Root.tsx：分辨率、帧率、总时长。改变分镜总长时同时更新根合成时长及渲染检查帧上限。
- scripts/generate-assets.py：可重生成灰度合成纹理，需 Python + numpy + scipy + Pillow；现成素材已包含在 public/phantom，不必运行科研代码。
- 加配音：把已获授权且对齐 115 秒节奏的音频放到 public/voiceover.wav，Studio 中将 voiceover 属性改为 voiceover.wav。命令行先设置 $env:VOICEOVER='voiceover.wav' 再 npm run render。音频时长改变时须调整分镜、字幕及总帧数。

## 交付文件
最终 MP4、联系表和视频元数据在 ../../outputs/cognitive-ultrasound-explainer。final-previews/ 是从实际 MP4 抽取的最终画面；previews/ 是渲染前的制作检查图。

Windows 下可运行 python scripts/verify-video.py，核对视频元数据、完整解码、两侧持帧差异并重建联系表（需 Pillow、numpy）。
分镜 docs/storyboard.md；旁白 docs/narration.md；字幕 docs/narration.srt；事实与素材说明 docs/sources.md；检查记录 docs/qa.md。

本制作不调用训练、推理、AutoDL 或付费云实验，不改科研算法、数据、实验配置。源码按仓库 Git 管理；视频、依赖和渲染缓存不提交。
