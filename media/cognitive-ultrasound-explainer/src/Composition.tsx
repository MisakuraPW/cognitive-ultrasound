import React from "react";
import { AssetGate } from "./AssetGate";
import { AbsoluteFill, useCurrentFrame, staticFile, Sequence } from "remotion";
import { Audio } from "@remotion/media";
import { TransitionSeries } from "@remotion/transitions";
import { C, SCENES, LINE_SETS, DENSE, clamp, smooth } from "./config";
import { Fan, T, Line } from "./visuals";
import { Captions } from "./Captions";
import { Pulse } from "./scenes/Pulse";
import { Build } from "./scenes/Build";
import { Tradeoff } from "./scenes/Tradeoff";
import { Reconstruct } from "./scenes/Reconstruct";
import { Uncertainty } from "./scenes/Uncertainty";
import { Closing } from "./scenes/Closing";
export type VideoProps = { voiceover: string };
const Overlay: React.FC<{ start: number }> = ({ start }) => {
  const frame = useCurrentFrame() + start * 30,
    t = frame / 30;
  return (
    <svg width="1920" height="1080" viewBox="0 0 1920 1080">
      <g
        opacity={Math.min(
          smooth((t - start) / 0.7),
          smooth((SCENES.find((s) => s.start === start)!.end - t) / 0.45),
        )}
      >
        {start === 0 ? (
          <Pulse t={t} />
        ) : start === 22 ? (
          <Build t={t} frame={frame} />
        ) : start === 38 ? (
          <Tradeoff t={t} frame={frame} />
        ) : start === 62 ? (
          <Reconstruct t={t} />
        ) : start === 80 ? (
          <Uncertainty t={t} />
        ) : (
          <Closing t={t} />
        )}
      </g>
    </svg>
  );
};
export const CognitiveUltrasound: React.FC<VideoProps> = ({ voiceover }) => {
  const frame = useCurrentFrame(),
    t = frame / 30,
    scene = Math.max(
      0,
      SCENES.findIndex((s) => t >= s.start && t < s.end),
    );
  let x = 620,
    y = 270,
    scale = 1,
    fp = frame,
    lines: number[] = [],
    only = false,
    estimate = 0,
    heat = 0,
    heatStage = 0,
    half = 48;
  if (t < 16) {
    fp = frame;
  } else if (t < 22) {
    fp = 0;
    only = true;
    lines = [12];
  } else if (t < 30) {
    fp = 0;
    only = true;
    lines = [
      12,
      ...Array.from(
        { length: Math.floor(clamp((t - 22) / 7) * 49) },
        (_, i) => -48 + i * 2,
      ),
    ];
  } else if (t < 38) {
    fp = Math.floor(frame / 5) * 5;
  } else if (t < 57) {
    const p = smooth((t - 38) / 1.2);
    x = 620 - 115 * p;
    y = 270 + 70 * p;
    scale = 1 - 0.3 * p;
    only = true;
    lines = DENSE.slice(
      0,
      t >= 47 ? 18 : Math.floor(clamp((t - 39) / 7.2) * 18),
    );
    fp = t >= 47 ? Math.floor((frame - 47 * 30) / 54) * 54 : 0;
  } else if (t < 62) {
    const p = smooth(t - 57);
    x = 505 + 115 * p;
    y = 340 - 70 * p;
    scale = 0.7 + 0.3 * p;
    half = 48 - 23 * smooth((t - 58) / 2);
    lines = DENSE.filter((a) => Math.abs(a) < half);
  } else if (t < 68) {
    half = 25 + 23 * smooth(t - 62);
    fp = 0;
    only = true;
    lines = [];
  } else if (t < 80) {
    fp = 0;
    only = true;
    lines = LINE_SETS[0].slice(0, Math.min(6, Math.floor((t - 68) * 2) + 1));
    estimate = smooth((t - 73) / 3);
  } else if (t < 105) {
    const st = t < 97 ? 0 : t < 101 ? 1 : 2;
    const since = t - (st === 1 ? 97 : 101);
    fp = st > 0 && since < 1 ? (st - 1) * 13 : st * 13;
    only = true;
    lines =
      st === 0
        ? LINE_SETS[st]
        : LINE_SETS[st].slice(0, Math.min(6, Math.floor(since * 7) + 1));
    estimate = 1;
    heat = smooth((t - 84) / 3) * (st === 0 ? 1 : st === 1 ? 0.85 : 0.92);
    heatStage = st > 0 && since < 1 ? (st === 2 ? 1 : 0) : st === 1 ? 1 : 0;
  } else {
    const p = smooth((t - 105) / 1.5);
    x = 620 - 60 * p;
    scale = 1 - 0.03 * p;
    only = true;
    lines = LINE_SETS[Math.floor((t - 105) / 3) % 3];
    estimate = 1;
    fp = Math.floor(frame / 4) * 4;
    heat = (1 - p) * 0.8;
  }
  return (
    <AbsoluteFill
      style={{
        background: C.bg,
        fontFamily: '"Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
      }}
    >
      <svg width="1920" height="1080" viewBox="0 0 1920 1080">
        <defs>
          <radialGradient id="ambient">
            <stop stopColor="#143340" stopOpacity=".5" />
            <stop offset="1" stopColor={C.bg} stopOpacity="0" />
          </radialGradient>
        </defs>
        <ellipse cx="610" cy="560" rx="650" ry="470" fill="url(#ambient)" />
        <path d="M96 137 H1824" stroke={C.line} strokeWidth="1" />
        <T x={100} y={71} size={25} color={C.cyan} weight={600}>
          认知超声 / COGNITIVE ULTRASOUND
        </T>
        <T x={100} y={118} size={39} weight={600}>
          {SCENES[scene].title}
        </T>
        <T x={1820} y={76} size={25} anchor="end" color={C.muted}>
          原理示意 · 全部素材合成
        </T>
        <Fan
          id="main"
          x={x}
          y={y}
          scale={scale}
          frame={fp}
          lines={lines}
          onlyLines={only}
          estimateOpacity={estimate}
          heat={heat}
          heatStage={heatStage}
          halfAngle={half}
          lineHalfWidth={t >= 38 && t < 57 ? 2.4 : 1.35}
        />
        {t >= 22 && t < 29 && (
          <g transform="translate(620 270)">
            <Line
              angle={-48 + Math.floor(clamp((t - 22) / 7) * 48) * 2}
              width={4}
            />
          </g>
        )}
        {t >= 73 && (
          <g>
            <path d="M1140 205 h35" stroke={C.cyan} strokeWidth="3" />
            <T x={1187} y={214} size={24} color={C.cyan}>
              实测
            </T>
            <path
              d="M1340 205 h35"
              stroke={C.purple}
              strokeWidth="3"
              strokeDasharray="6 5"
            />
            <T x={1387} y={214} size={24} color={C.purple}>
              估计
            </T>
            {t >= 84 && t < 105 && (
              <>
                <circle cx="1555" cy="205" r="7" fill={C.orange} />
                <T x={1573} y={214} size={24} color={C.orange}>
                  拿不准
                </T>
              </>
            )}
          </g>
        )}
        <path d="M96 912 H1824" stroke={C.line} />
        <T x={100} y={1042} size={23} color={C.muted}>
          传统逐线聚焦扫描的简化示意
          {t >= 38 && t < 62 ? " · 线数与时间非实测数据" : ""}
        </T>
        <T x={1820} y={1042} size={23} anchor="end" color={C.muted}>
          {String(scene + 1).padStart(2, "0")} / 06
        </T>
        <rect
          x="100"
          y="1066"
          width={(1720 * frame) / 3449}
          height="3"
          fill={C.cyan}
          opacity=".6"
        />
      </svg>
      <TransitionSeries>
        {SCENES.map((s) => (
          <TransitionSeries.Sequence
            key={s.start}
            name={s.title}
            durationInFrames={(s.end - s.start) * 30}
          >
            <Overlay start={s.start} />
          </TransitionSeries.Sequence>
        ))}
      </TransitionSeries>
      <AssetGate />
      <Captions frame={frame} />
      {voiceover ? (
        <Sequence name="可选中文配音">
          <Audio src={staticFile(voiceover)} />
        </Sequence>
      ) : null}
    </AbsoluteFill>
  );
};
