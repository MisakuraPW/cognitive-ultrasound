import React from "react";
import { C, smooth } from "../config";
import { Fan, T, Arrow } from "../visuals";
export const Build: React.FC<{ t: number; frame: number }> = ({ t, frame }) => (
  <g>
    {t < 30 ? (
      <g opacity={smooth(t - 22)}>
        <T x={1170} y={380} size={52} weight={600}>
          逐条采集
        </T>
        <T x={1170} y={462} size={52} weight={600}>
          逐步成像
        </T>
        <path d="M1170 522 h510" stroke={C.line} />
        <T x={1170} y={590} color={C.muted}>
          扫描方向从探头向外发散
        </T>
        <T x={1170} y={648} color={C.cyan}>
          每个方向，都经历发射与接收
        </T>
      </g>
    ) : (
      <g opacity={smooth(t - 30)}>
        <T x={1190} y={310} size={48} weight={600}>
          一帧接着一帧
        </T>
        {[0, 1, 2].map((i) => (
          <g key={i}>
            <Fan
              id={`strip-${i}`}
              x={1220 + i * 220}
              y={425}
              scale={0.23}
              frame={frame - 18 * (2 - i)}
              probe={false}
            />
            <T
              x={1220 + i * 220}
              y={615}
              anchor="middle"
              size={28}
              color={C.muted}
            >
              {["刚才", "随后", "现在"][i]}
            </T>
          </g>
        ))}
        <Arrow x={1180} y={685} length={565} />
        <T x={1465} y={760} size={34} anchor="middle" color={C.cyan}>
          连续更新 → 看见运动
        </T>
      </g>
    )}
  </g>
);
