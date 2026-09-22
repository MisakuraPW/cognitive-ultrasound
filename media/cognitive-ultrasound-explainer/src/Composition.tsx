import { AbsoluteFill, useCurrentFrame } from "remotion";
import { AssetGate } from "./AssetGate";
import { C, MAIN, DENSE, LINE_SETS, TRADE, smooth, clamp } from "./config";
import { Fan } from "./visuals";
import { Pulse } from "./scenes/Pulse";
import { Build } from "./scenes/Build";
import { Tradeoff } from "./scenes/Tradeoff";
import { Reconstruct } from "./scenes/Reconstruct";
import { Uncertainty } from "./scenes/Uncertainty";
export const CognitiveUltrasound = () => {
  const frame = useCurrentFrame(),
    t = frame / 30;
  let x = MAIN.x,
    y = MAIN.y,
    scale = MAIN.scale,
    fp = 0,
    efp = 0,
    lines: number[] = [],
    only = true,
    context = 0,
    estimate = 0,
    heat = 0,
    heatStage = 0,
    lineOpacity = 1,
    halfWidth = 1.7;
  if (t < 5) {
    context = 0.36 * (1 - smooth((t - 3.9) / 0.8));
    lines = t >= 3.9 ? [12] : [];
  } else if (t < 9.5) {
    lines = [
      12,
      ...DENSE.slice(0, Math.floor(clamp((t - 5) / 4.5) * 24)),
    ].filter((v, i, a) => a.indexOf(v) === i);
    halfWidth = 2.3;
  } else if (t < 12) {
    only = false;
    fp = Math.floor(frame / 3) * 3;
  } else if (t < 24) {
    const p = smooth((t - 12) / 1.1);
    x = MAIN.x + (510 - MAIN.x) * p;
    y = MAIN.y + (265 - MAIN.y) * p;
    scale = MAIN.scale + (1.02 - MAIN.scale) * p;
    lines = DENSE;
    lineOpacity = 0.55;
    halfWidth = 2.3;
    const local = Math.max(0, frame - 390);
    fp = 390 + Math.floor(local / TRADE.slowPeriod) * TRADE.slowPeriod;
  } else if (t < 25.5) {
    const p = smooth((t - 24) / 1.2);
    x = 510 + (MAIN.x - 510) * p;
    y = 265 + (MAIN.y - 265) * p;
    scale = 1.02 + (MAIN.scale - 1.02) * p;
    context = 0.7 * (1 - p);
  } else if (t < 37) {
    lines = LINE_SETS[0].slice(
      0,
      Math.min(6, Math.floor((t - 25.5) / 0.5) + 1),
    );
    estimate = smooth((t - 30) / 4);
  } else {
    const stage = t < 42 ? 0 : t < 48.2 ? 1 : 2;
    const start = stage === 1 ? 42 : 48.2,
      dt = t - start;
    fp = stage * 14;
    efp = fp;
    lines =
      stage === 0
        ? LINE_SETS[0]
        : LINE_SETS[stage].slice(0, Math.min(6, Math.floor(dt / 0.25) + 1));
    estimate = stage === 0 ? 1 : smooth((dt - 1.8) / 1.7);
    if (stage === 0) heat = smooth((t - 37.7) / 1);
    if (stage === 1) {
      heat = smooth((t - 45.6) / 0.6) * 0.88;
      heatStage = 1;
    }
    if (stage === 2) {
      heat = smooth((t - 51.7) / 0.6) * 0.72;
    }
    // Last two seconds are a literal held image, not another page or fade-out.
  }
  return (
    <AbsoluteFill
      style={{
        background: C.bg,
        fontFamily: '"Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
      }}
    >
      <svg width="1920" height="1080" viewBox="0 0 1920 1080">
        <Fan
          id="main"
          x={x}
          y={y}
          scale={scale}
          frame={fp}
          estimateFrame={efp}
          lines={lines}
          onlyLines={only}
          contextOpacity={context}
          estimateOpacity={estimate}
          heat={heat}
          heatStage={heatStage}
          lineHalfWidth={halfWidth}
          lineOpacity={lineOpacity}
        />
        {t < 5 && <Pulse t={t} />}
        {t >= 5 && t < 12 && <Build t={t} />}
        {t >= 12 && t < 24 && <Tradeoff t={t} frame={frame} />}
        {t >= 24 && <Reconstruct t={t} />}
        {t >= 37 && <Uncertainty t={t} />}
      </svg>
      <AssetGate />
    </AbsoluteFill>
  );
};
