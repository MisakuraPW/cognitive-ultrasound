import React from "react";
import { C, MAIN, LINE_SETS, smooth, clamp } from "../config";
import { T, Arrow, Line, point } from "../visuals";
export const Uncertainty: React.FC<{ t: number }> = ({ t }) => {
  const second = t >= 45.6,
    candidate = t >= 39.3 && t < 42 ? 1 : t >= 46.2 && t < 48.2 ? 2 : 0;
  const cOpacity =
    candidate === 1 ? smooth((t - 39.3) / 0.8) : smooth((t - 46.2) / 0.6);
  const heatAlpha =
    t < 42
      ? smooth((t - 37.7) / 1)
      : t < 45.6
        ? 0
        : t < 48.2
          ? smooth((t - 45.6) / 0.6)
          : t < 51.7
            ? 0
            : 0.65;
  const hx = second && t < 48.2 ? -76 : 100,
    hy = second && t < 48.2 ? 330 : 357;
  const left = second && t < 48.2,
    tx = left ? 210 : 1450,
    ty = left ? 650 : 650;
  const cursor =
    t >= 41.5 && t < 43
      ? clamp((t - 41.5) / 0.5)
      : t >= 47.7 && t < 49.2
        ? clamp((t - 47.7) / 0.5)
        : 0;
  return (
    <g>
      <g opacity={heatAlpha}>
        <T x={tx} y={ty} size={40} color={C.orange}>
          重建不确定性
        </T>
        <path
          d={`M${left ? tx + 235 : tx} ${ty + 12} L${MAIN.x + hx * MAIN.scale} ${MAIN.y + hy * MAIN.scale}`}
          fill="none"
          stroke={C.orange}
          strokeWidth="2.5"
        />
      </g>
      {candidate > 0 && (
        <g opacity={cOpacity}>
          <T x={1240} y={315} size={36} color={C.orange}>
            候选方向
          </T>
          <g transform={`translate(${MAIN.x} ${MAIN.y}) scale(${MAIN.scale})`}>
            {LINE_SETS[candidate].map((a) => (
              <Line
                key={a}
                angle={a}
                color={C.orange}
                dash
                width={3.2}
                opacity={
                  (candidate === 1 ? a >= 8 && a <= 22 : a >= -23 && a <= -2)
                    ? 1
                    : 0.36
                }
              />
            ))}
            {LINE_SETS[candidate]
              .filter((a) =>
                candidate === 1 ? a >= 8 && a <= 22 : a >= -23 && a <= -2,
              )
              .map((a) => {
                const p = point(a, candidate === 1 ? 370 : 350);
                return (
                  <circle
                    key={a}
                    cx={p[0]}
                    cy={p[1]}
                    r="7"
                    fill="#FFD18C"
                    stroke={C.orange}
                    strokeWidth="2"
                  />
                );
              })}
          </g>
        </g>
      )}
      <g opacity={smooth((t - 37) / 0.5)}>
        <T x={765} y={1020} anchor="middle" size={44}>
          当前帧
        </T>
        <Arrow x={890} y={1003} length={145} color={C.cyan} />
        <T
          x={1155}
          y={1020}
          anchor="middle"
          size={44}
          color={cursor > 0.5 ? C.cyan : C.text}
        >
          下一帧
        </T>
        {cursor > 0 && (
          <circle cx={890 + 145 * cursor} cy="1003" r="7" fill={C.cyan} />
        )}
      </g>
    </g>
  );
};
