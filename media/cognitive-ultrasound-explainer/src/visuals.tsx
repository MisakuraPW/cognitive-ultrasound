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
  size = 32,
  color = C.text,
  anchor = "start",
  weight = 400,
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
  color?: string;
  dash?: boolean;
  width?: number;
  opacity?: number;
}> = ({ angle, color = C.cyan, dash = false, width = 2.3, opacity = 1 }) => {
  const p = point(angle, 553),
    s = point(angle, 22);
  return (
    <path
      d={`M${s[0]} ${s[1]} L${p[0]} ${p[1]}`}
      stroke={color}
      strokeWidth={width}
      strokeDasharray={dash ? "9 9" : undefined}
      opacity={opacity}
    />
  );
};
export const Probe = () => (
  <g>
    <path
      d="M-17 -83 C-13 -105 22 -104 24 -82 L19 -33 Q0 -22 -20 -33 Z"
      fill="#bccbd0"
    />
    <path d="M-20 -33 Q0 -25 19 -33 L21 -10 Q0 2 -23 -10 Z" fill="#4e727e" />
    <path
      d="M3 -101 C5 -117 34 -107 39 -124"
      fill="none"
      stroke="#597580"
      strokeWidth="7"
    />
    <path d="M-16 -5 Q0 1 16 -5" fill="none" stroke={C.cyan} strokeWidth="4" />
  </g>
);
export const Fan: React.FC<{
  id: string;
  x?: number;
  y?: number;
  scale?: number;
  frame?: number;
  mode?: "observed" | "estimate" | "possible";
  variant?: number;
  reveal?: number;
  lines?: number[];
  onlyLines?: boolean;
  estimateOpacity?: number;
  halfAngle?: number;
  heat?: number;
  heatStage?: number;
  probe?: boolean;
  lineHalfWidth?: number;
}> = ({
  id,
  x = 620,
  y = 255,
  scale = 1,
  frame = 0,
  mode = "observed",
  variant = 0,
  reveal = 48,
  lines = [],
  onlyLines = false,
  estimateOpacity = 0,
  halfAngle = 48,
  heat = 0,
  heatStage = 0,
  probe = true,
  lineHalfWidth = 1.35,
}) => {
  const src =
    mode === "possible"
      ? `possible-${variant}.png`
      : `${mode}-${String(phase(frame)).padStart(2, "0")}.png`;
  const hx = heatStage === 1 ? -76 : 100,
    hy = heatStage === 1 ? 330 : 357;
  return (
    <g transform={`translate(${x} ${y}) scale(${scale})`}>
      <defs>
        <clipPath id={`${id}-fan`}>
          <path d={sector(-halfAngle, Math.min(reveal, halfAngle))} />
        </clipPath>
        <clipPath id={`${id}-lines`}>
          {lines.map((a, i) => (
            <path key={i} d={sector(a - lineHalfWidth, a + lineHalfWidth)} />
          ))}
        </clipPath>
        <radialGradient id={`${id}-heat`}>
          <stop stopColor={C.orange} stopOpacity=".66" />
          <stop offset=".55" stopColor={C.orange} stopOpacity=".25" />
          <stop offset="1" stopColor={C.orange} stopOpacity="0" />
        </radialGradient>
      </defs>
      <path
        d={sector(-halfAngle, halfAngle)}
        fill="#040b10"
        stroke={C.line}
        strokeWidth="2"
      />
      <g clipPath={`url(#${id}-fan)`}>
        {estimateOpacity > 0 && (
          <g opacity={estimateOpacity}>
            <image
              preserveAspectRatio="none"
              href={staticFile(
                `phantom/estimate-${String(phase(frame)).padStart(2, "0")}.png`,
              )}
              x="-500"
              y="0"
              width="1000"
              height="600"
            />
            <path d={sector()} fill={C.purple} opacity=".10" />
          </g>
        )}
        <g clipPath={onlyLines ? `url(#${id}-lines)` : undefined}>
          <image
            preserveAspectRatio="none"
            href={staticFile(`phantom/${src}`)}
            x="-500"
            y="0"
            width="1000"
            height="600"
          />
        </g>
        {heat > 0 && (
          <g opacity={heat}>
            <ellipse
              cx={hx}
              cy={hy}
              rx="113"
              ry="130"
              fill={`url(#${id}-heat)`}
            />
            <ellipse
              cx={-hx * 0.9}
              cy="455"
              rx="77"
              ry="58"
              fill={`url(#${id}-heat)`}
              opacity={heatStage === 1 ? 0.75 : 0.45}
            />
            <ellipse
              cx={hx}
              cy={hy}
              rx="76"
              ry="88"
              fill="none"
              stroke={C.orange}
              strokeDasharray="5 8"
              opacity=".55"
            />
          </g>
        )}
      </g>
      {estimateOpacity > 0 && (
        <path
          d={sector(-halfAngle, halfAngle)}
          fill="none"
          stroke={C.purple}
          strokeWidth="2.5"
          strokeDasharray="10 8"
          opacity={estimateOpacity}
        />
      )}
      {lines.map((a, i) => (
        <Line angle={a} key={i} opacity={0.8} width={2} />
      ))}
      {probe && <Probe />}
    </g>
  );
};
export const Label: React.FC<{
  x: number;
  y: number;
  color?: string;
  children: React.ReactNode;
  dashed?: boolean;
}> = ({ x, y, color = C.cyan, children, dashed }) => (
  <g>
    <path
      d={`M${x} ${y - 10} h42`}
      stroke={color}
      strokeWidth="4"
      strokeDasharray={dashed ? "7 6" : undefined}
    />
    <T x={x + 57} y={y} size={28} color={color}>
      {children}
    </T>
  </g>
);
export const Arrow: React.FC<{
  x: number;
  y: number;
  length?: number;
  color?: string;
}> = ({ x, y, length = 90, color = C.muted }) => (
  <path
    d={`M${x} ${y} h${length} m-12 -8 l12 8 l-12 8`}
    fill="none"
    stroke={color}
    strokeWidth="2.5"
  />
);
