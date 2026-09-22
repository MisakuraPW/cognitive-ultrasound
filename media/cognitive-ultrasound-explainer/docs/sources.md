# 素材、科学事实与边界
## 主要依据
1. W. L. van Nierop, O. Nolan, T. S. W. Stevens, R. J. G. van Sloun, *Patient-Adaptive Echocardiography Using Cognitive Ultrasound*, IEEE TMI 45(7), 4034–4046, 2026. DOI: [10.1109/TMI.2026.3691009](https://doi.org/10.1109/TMI.2026.3691009)。本地文件：G:/科研项目/毕设/文献/Patient-Adaptive_Echocardiography_Using_Cognitive_Ultrasound.pdf。查看引言、图 1、图 2、III-A/B：聚焦采集的范围/时间取舍，稀疏观测重建，当前后验指导后续采集。没有采用论文性能数字。
2. 本仓库 docs/CASL_architecture.md；vendor/casl/ulsa/agent.py 的 recover（约 610–659 行）：measurement_buffer → posterior_sample → 当前帧 posterior → action_selection → 下一 mask 写入状态。画面严格跨帧更新位置。vendor/casl/ulsa/selection.py 只用来核对策略接口；不把其他策略当 CASL 主策略。官方主策略是基于后验样本的熵与贪心重权，不是简单方差 top-k。本片“意见不一致”仅是外行理解不确定性的直觉，无公式或策略复现。
3. [BMUS 超声物理教学](https://www.bmus.org/media/resources/files/09.45EC.pdf)：短脉冲发射与回波接收、往返时间与深度、处理后的回声强度映射为图像。参考其事实，未使用原图。
4. [BMUS Ultrasound Interactions](https://www.bmus.org/education-and-cpd/ultrasound-physics-and-equipment/ultrasound-interactions/)：反射、散射、脉冲回波成像与距离定位。
5. 本地《基于自适应采样及图像生成的感知型超声方法研究.pdf》仅阅读摘要和背景。其 legacy LDM、动态信念滤波、固定 10+4 与本片 CASL 分开，不引入其算法或结果。
6. 整理/认知超声_预测式主动采集研究主线.md 与 整理/审计后的新主线.md 仅核对研究定位，不把未验证预测方法作为既有能力。
## 全部视觉素材均为合成原理示意
- scripts/generate-assets.py 使用解析心腔形状、平滑形变和固定种子散斑生成 48 相位灰度纹理。不是患者影像，不是物理声场仿真，不是实验或推理结果。
- observed-* 是教学场景的合成观测素材；estimate-* 是分别扰动的示意重建。已测方向覆盖在估计之上。possible-* 用于表示几种可能形态；它们不是 CASL 后验输出。热图是手工布置的解释图，不是计算得到的熵或真实误差。
- 四腔切面是简化形态，不用于解剖识别或诊断。探头、声波、时间轴、标记全部自行用 SVG 绘制。
- 未复用论文图片、官方视频、患者资料或第三方音乐。论文原图只在本地 reference-check 下用于阅读，不在成片或源码内分发。
- 使用系统 Microsoft YaHei；不复制、分发字体文件。其他电脑需安装中文字体。
## 科学边界
全片限定传统逐线聚焦扫描的简化解释。线密度是空间采样程度，不能直接等同真实空间分辨率。影片固定 30 fps 与超声采集更新率不同。18/6 条与时间格仅为教学比例，无实际 FPS 或提速倍数。下一帧示意始终 6 条，不宣称每帧固定线加额外线。速度收益仍取决于计算和设备实现。
