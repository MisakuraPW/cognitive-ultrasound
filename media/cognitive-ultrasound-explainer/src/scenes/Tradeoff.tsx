import React from "react";
import { C, clamp, smooth, SPARSE } from "../config";
import { Fan, T } from "../visuals";
const Slots: React.FC<{ x: number; y: number; n: number; elapsed: number }> = ({
  x,
  y,
  n,
  elapsed,
}) => (
  <g>
    {Array.from({ length: 18 }, (_, i) => (
      <g key={i} opacity={i < n ? 1 : 0.15}>
        <rect
          x={x + i * 31}
          y={y}
          width="27"
          height="24"
          rx="3"
          fill={i < elapsed ? C.cyan : C.line}
        />
        <rect
          x={x + i * 31}
          y={y}
          width="5"
          height="24"
          fill={i < elapsed ? "#dffaff" : "#4a6370"}
        />
      </g>
    ))}
    <T x={x} y={y + 65} size={27} color={C.muted}>
      每格：发射 + 等待回波
    </T>
  </g>
);
export const Tradeoff: React.FC<{ t: number; frame: number }> = ({
  t,
  frame,
}) => {
  const local = frame - 38 * 30,
    motion = t >= 47,
    progress = clamp((t - 39) / 7.2) * 18;
  const rightFrame = motion ? Math.floor((local - 270) / 18) * 18 : 0;
  return (
    <g>
      {t < 57 ? (
        <g opacity={smooth((t - 39.2) / 0.6)}>
          <Fan
            id="compare-right"
            x={1405}
            y={340}
            scale={0.7}
            frame={rightFrame}
            lines={SPARSE.slice(0, motion ? 6 : Math.floor(progress))}
            onlyLines
            lineHalfWidth={2.4}
          />
          <T x={505} y={236} size={43} anchor="middle" weight={600}>
            采得密
          </T>
          <T x={1405} y={236} size={43} anchor="middle" weight={600}>
            采得疏
          </T>
          <T x={960} y={185} anchor="middle" size={28} color={C.muted}>
            相同深度 · 相同扇形范围{motion ? " · 同一运动时钟" : ""}
          </T>
          <Slots
            x={228}
            y={802}
            n={18}
            elapsed={motion ? ((local - 270) % 54) / 3 : progress}
          />
          <Slots
            x={1128}
            y={802}
            n={6}
            elapsed={motion ? ((local - 270) % 18) / 3 : progress}
          />
          <T x={505} y={765} size={30} anchor="middle" color={C.cyan}>
            {motion
              ? "保持上一帧，直到新一帧采完"
              : "更多方向 → 更长的采集时间"}
          </T>
          <T x={1405} y={765} size={30} anchor="middle" color={C.cyan}>
            {motion
              ? "较早更新，测到的信息较少"
              : progress >= 6
                ? "这一帧先采完了"
                : "较少方向 → 较早采完"}
          </T>
        </g>
      ) : (
        <g opacity={smooth(t - 57)}>
          <T x={1170} y={360} size={50} weight={600}>
            范围也能换时间
          </T>
          <T x={1170} y={455} size={34}>
            保持近似的角度采样间隔
          </T>
          <T x={1170} y={515} size={34} color={C.cyan}>
            范围变窄 → 要采的线变少
          </T>
          <T x={1170} y={632} size={34} color={C.muted}>
            代价：看见的区域变小
          </T>
          <T x={1170} y={755} size={27} color={C.muted}>
            还受深度、聚焦和系统处理等因素影响
          </T>
        </g>
      )}
    </g>
  );
};
