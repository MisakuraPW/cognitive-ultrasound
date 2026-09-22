import React from "react";
import { C, MAIN, LINE_SETS, smooth } from "../config";
import { T, point } from "../visuals";
export const Reconstruct: React.FC<{ t: number }> = ({ t }) => {
  const stage = t < 42 ? 0 : t < 48.2 ? 1 : 2;
  const p = point(LINE_SETS[stage][0], 330);
  const estimateLabel =
    stage === 0
      ? smooth((t - 30) / 1)
      : smooth((t - (stage === 1 ? 42 : 48.2) - 1.8) / 0.7);
  return (
    <g>
      <g opacity={smooth((t - 25.5) / 0.5)}>
        <T x={410} y={485} size={44} color={C.cyan}>
          实测
        </T>
        <path
          d={`M500 498 L${MAIN.x + p[0] * MAIN.scale} ${MAIN.y + p[1] * MAIN.scale}`}
          fill="none"
          stroke={C.cyan}
          strokeWidth="3"
        />
      </g>
      <g opacity={estimateLabel}>
        <T x={1480} y={760} size={44} color={C.purple}>
          重建估计
        </T>
        <path
          d="M1480 773 L1360 815"
          fill="none"
          stroke={C.purple}
          strokeWidth="3"
          strokeDasharray="9 7"
        />
      </g>
    </g>
  );
};
