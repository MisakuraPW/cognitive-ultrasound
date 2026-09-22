import React from "react";
import { C, MAIN, smooth } from "../config";
import { T, point, sector } from "../visuals";
export const Pulse: React.FC<{ t: number }> = ({ t }) => {
  const travel = (t - 0.35) * 260,
    a = 12;
  return (
    <g opacity={1 - smooth((t - 4.7) / 0.3)}>
      <g transform={`translate(${MAIN.x} ${MAIN.y}) scale(${MAIN.scale})`}>
        <path d={sector(a - 3, a + 3)} fill={C.cyanLight} opacity=".08" />
        {[150, 310, 455].map((d) => {
          const p = point(a, d);
          return (
            <ellipse
              key={d}
              cx={p[0]}
              cy={p[1]}
              rx="23"
              ry="5"
              fill="#D5E4E8"
            />
          );
        })}
        {travel > 0 &&
          travel < 555 &&
          [0, 10, 20].map((off) => {
            const p = point(a, travel - off);
            return (
              <path
                key={off}
                d={`M${p[0] - 25} ${p[1]} Q${p[0]} ${p[1] + 10} ${p[0] + 25} ${p[1] - 5}`}
                fill="none"
                stroke={C.cyanLight}
                strokeWidth="4"
              />
            );
          })}
        {[150, 310, 455].map((d) => {
          const r = 2 * d - travel;
          if (travel < d || r < 7) return null;
          const p = point(a, r);
          return (
            <path
              key={d}
              d={`M${p[0] - 23} ${p[1]} Q${p[0]} ${p[1] - 12} ${p[0] + 23} ${p[1] + 3}`}
              fill="none"
              stroke="#FFFFFF"
              strokeWidth="4"
            />
          );
        })}
      </g>
      <T x={1080} y={180} size={43}>
        发射 / 接收
      </T>
      <T x={960} y={1007} anchor="middle" size={32} color={C.muted}>
        传播慢放示意
      </T>
    </g>
  );
};
