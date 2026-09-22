import React from "react";
import { staticFile } from "remotion";
import { C, phase } from "./config";
export const point = (a: number, r: number) => [
  Math.sin((a * Math.PI) / 180) * r,
  Math.cos((a * Math.PI) / 180) * r,
];
export const sector = (a = -48, b = 48, r = 555) => {
  const p = point(a, r),
    q = point(b, r);
  return `M0 0 L${p[0]} ${p[1]} A${r} ${r} 0 0 0 ${q[0]} ${q[1]} Z`;
};
export const arc = (a: number, b: number, r: number) => {
  const p = point(a, r),
    q = point(b, r);
  return `M${p[0]} ${p[1]} A${r} ${r} 0 0 0 ${q[0]} ${q[1]}`;
};
export const T: React.FC<{
  x: number;
  y: number;
  children: React.ReactNode;
  size?: number;
  color?: string;
  anchor?: "start" | "middle" | "end";
  weight?: number;
}> = ({
  x,
  y,
  children,
  size = 40,
  color = C.text,
  anchor = "start",
  weight = 500,
}) => (
  <text
    x={x}
    y={y}
    fontSize={size}
    fill={color}
    textAnchor={anchor}
    fontWeight={weight}
  >
    {children}
  </text>
);
export const Line: React.FC<{
  angle: number;
  radius?: number;
  color?: string;
  dash?: boolean;
  width?: number;
  opacity?: number;
}> = ({
  angle,
  radius = 553,
  color = C.cyan,
  dash = false,
  width = 2.8,
  opacity = 1,
}) => {
  const p = point(angle, radius),
    s = point(angle, 23),
    d = `M${s[0]} ${s[1]} L${p[0]} ${p[1]}`;
  return (
    <g opacity={opacity}>
      <path
        d={d}
        fill="none"
        stroke={dash ? "#FFF2D8" : "#BDEFF5"}
        strokeWidth={width + 1.7}
        strokeDasharray={dash ? "11 9" : undefined}
        opacity=".8"
      />
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={width}
        strokeDasharray={dash ? "11 9" : undefined}
      />
    </g>
  );
};
export const Probe = () => (
  <g>
    <path
      d="M-17 -83 C-13 -105 22 -104 24 -82 L19 -33 Q0 -22 -20 -33 Z"
      fill="#AABCC3"
      stroke="#607D88"
      strokeWidth="1.8"
    />
    <path d="M-20 -33 Q0 -25 19 -33 L21 -10 Q0 2 -23 -10 Z" fill="#42626F" />
    <path
      d="M3 -101 C5 -117 34 -107 39 -124"
      fill="none"
      stroke="#607D88"
      strokeWidth="6"
    />
    <path d="M-16 -5 Q0 1 16 -5" fill="none" stroke={C.cyan} strokeWidth="4" />
  </g>
);
type FanProps = {
  id: string;
  x?: number;
  y?: number;
  scale?: number;
  frame?: number;
  estimateFrame?: number;
  radius?: number;
  lines?: number[];
  onlyLines?: boolean;
  estimateOpacity?: number;
  contextOpacity?: number;
  observedOpacity?: number;
  halfAngle?: number;
  heat?: number;
  heatStage?: number;
  lineHalfWidth?: number;
  lineOpacity?: number;
  probe?: boolean;
};
export const Fan: React.FC<FanProps> = ({
  id,
  x = 960,
  y = 225,
  scale = 1.27,
  frame = 0,
  estimateFrame = frame,
  radius = 555,
  lines = [],
  onlyLines = false,
  estimateOpacity = 0,
  contextOpacity = 0,
  observedOpacity = 1,
  halfAngle = 48,
  heat = 0,
  heatStage = 0,
  lineHalfWidth = 1.7,
  lineOpacity = 1,
  probe = true,
}) => {
  const observed = staticFile(
    `phantom/observed-${String(phase(frame)).padStart(2, "0")}.png`,
  );
  const estimated = staticFile(
    `phantom/estimate-${String(phase(estimateFrame)).padStart(2, "0")}.png`,
  );
  const hx = heatStage === 1 ? -76 : 100,
    hy = heatStage === 1 ? 330 : 357;
  return (
    <g transform={`translate(${x} ${y}) scale(${scale})`}>
      <defs>
        <clipPath id={`${id}-fan`}>
          <path d={sector(-halfAngle, halfAngle, radius)} />
        </clipPath>
        <clipPath id={`${id}-lines`}>
          {lines.map((a) => (
            <path
              key={a}
              d={sector(a - lineHalfWidth, a + lineHalfWidth, radius)}
            />
          ))}
        </clipPath>
        <radialGradient id={`${id}-heat`}>
          <stop stopColor={C.heat} stopOpacity=".76" />
          <stop offset=".6" stopColor={C.heat} stopOpacity=".35" />
          <stop offset="1" stopColor={C.heat} stopOpacity="0" />
        </radialGradient>
      </defs>
      <path
        d={sector(-halfAngle, halfAngle, radius)}
        fill="#060A0D"
        stroke="#B8C6CD"
        strokeWidth="1.5"
      />
      <g clipPath={`url(#${id}-fan)`}>
        {contextOpacity > 0 && (
          <image
            href={observed}
            x="-500"
            y="0"
            width="1000"
            height="600"
            preserveAspectRatio="none"
            opacity={contextOpacity}
          />
        )}
        {estimateOpacity > 0 && (
          <image
            href={estimated}
            x="-500"
            y="0"
            width="1000"
            height="600"
            preserveAspectRatio="none"
            opacity={estimateOpacity}
          />
        )}
        <g
          clipPath={onlyLines ? `url(#${id}-lines)` : undefined}
          opacity={observedOpacity}
        >
          <image
            href={observed}
            x="-500"
            y="0"
            width="1000"
            height="600"
            preserveAspectRatio="none"
          />
        </g>
        {heat > 0 && (
          <g opacity={heat}>
            <ellipse
              cx={hx}
              cy={hy}
              rx="100"
              ry="112"
              fill={`url(#${id}-heat)`}
            />
            <ellipse
              cx={-hx * 0.9}
              cy="450"
              rx="70"
              ry="55"
              fill={`url(#${id}-heat)`}
              opacity={heatStage === 1 ? 0.7 : 0.5}
            />
            <ellipse
              cx={hx}
              cy={hy}
              rx="63"
              ry="76"
              fill="none"
              stroke="#FFD091"
              strokeWidth="2"
              strokeDasharray="6 7"
              opacity=".7"
            />
          </g>
        )}
      </g>
      {estimateOpacity > 0 && (
        <g opacity={estimateOpacity}>
          <path
            d={sector(-halfAngle, halfAngle, radius)}
            fill="none"
            stroke="#EAE0FE"
            strokeWidth="5"
          />
          <path
            d={sector(-halfAngle, halfAngle, radius)}
            fill="none"
            stroke={C.purple}
            strokeWidth="3.3"
            strokeDasharray="10 8"
          />
        </g>
      )}
      {lines.map((a) => (
        <Line key={a} angle={a} radius={radius - 2} opacity={lineOpacity} />
      ))}
      {probe && <Probe />}
    </g>
  );
};
export const Arrow: React.FC<{
  x: number;
  y: number;
  length?: number;
  color?: string;
}> = ({ x, y, length = 130, color = C.muted }) => (
  <path
    d={`M${x} ${y} h${length} m-14 -10 l14 10 l-14 10`}
    fill="none"
    stroke={color}
    strokeWidth="3"
  />
);
