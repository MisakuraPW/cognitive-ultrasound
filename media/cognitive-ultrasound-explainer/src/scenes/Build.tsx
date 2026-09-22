import React from "react";
import { C, MAIN, DENSE, smooth } from "../config";
import { T, point } from "../visuals";
export const Build: React.FC<{ t: number }> = ({ t }) => {
  const p = ((t - 5) / 4.5) * 24,
    i = Math.min(23, Math.floor(p)),
    within = p - i,
    a = DENSE[Math.max(0, i)],
    r = within < 0.5 ? within * 2 * 540 : (1 - within) * 2 * 540,
    pos = point(a, r);
  return (
    <g opacity={smooth((t - 5) / 0.3) * (1 - smooth((t - 11.5) / 0.5))}>
      <T x={1080} y={180} size={43}>
        逐线成像
      </T>
      {t < 9.5 && (
        <g transform={`translate(${MAIN.x} ${MAIN.y}) scale(${MAIN.scale})`}>
          <path
            d={`M${pos[0] - 14} ${pos[1]} Q${pos[0]} ${pos[1] + 8} ${pos[0] + 14} ${pos[1]}`}
            fill="none"
            stroke={within < 0.5 ? C.cyanLight : "#FFFFFF"}
            strokeWidth="4"
          />
        </g>
      )}
    </g>
  );
};
