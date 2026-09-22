import React from "react";
import { C, SPARSE, TRADE, smooth, clamp } from "../config";
import { Fan, T, arc, point } from "../visuals";
const Dimension: React.FC<{
  x: number;
  half: number;
  radius: number;
  right?: boolean;
}> = ({ x, half, radius, right = false }) => {
  const a = right ? half + 9 : -half - 9,
    p = point(a, 40),
    q = point(a, radius);
  return (
    <g transform={`translate(${x} 265) scale(1.02)`}>
      <path
        d={arc(-half, half, 135)}
        fill="none"
        stroke={C.cyanLight}
        strokeWidth="3"
      />
      <path
        d={`M${p[0]} ${p[1]} L${q[0]} ${q[1]}`}
        stroke={C.muted}
        strokeWidth="2.5"
      />
      {[p, q].map((v, i) => (
        <path
          key={i}
          d={`M${v[0] - 8} ${v[1] - 5} l16 10`}
          stroke={C.muted}
          strokeWidth="3"
        />
      ))}
    </g>
  );
};
export const Tradeoff: React.FC<{ t: number; frame: number }> = ({
  t,
  frame,
}) => {
  const local = Math.max(0, frame - TRADE.start * 30),
    p = smooth((t - 12.8) / 0.7) * (1 - smooth((t - 23.4) / 0.6));
  return (
    <g opacity={p}>
      <Fan
        id="trade-right"
        x={1410}
        y={265}
        scale={1.02}
        radius={TRADE.shallowRadius}
        halfAngle={33}
        frame={390 + Math.floor(local / TRADE.fastPeriod) * TRADE.fastPeriod}
        lines={SPARSE}
        onlyLines
        lineHalfWidth={2.1}
        lineOpacity={0.7}
      />
      <Dimension x={510} half={48} radius={555} />
      <Dimension x={1410} half={33} radius={435} right />
      <T x={960} y={78} anchor="middle" size={43}>
        看多广、看多深、采多密，都要花时间
      </T>
      <T x={710} y={390} size={36}>
        范围
      </T>
      <T x={1600} y={390} size={36}>
        范围
      </T>
      <T x={120} y={550} size={36}>
        深度
      </T>
      <T x={1735} y={550} size={36}>
        深度
      </T>
      <T x={980} y={570} anchor="middle" size={37}>
        采样密度
      </T>
      <path
        d="M900 586 L812 624 M1060 586 L1185 624"
        fill="none"
        stroke={C.muted}
        strokeWidth="2"
      />
      <T x={510} y={905} anchor="middle" size={35}>
        一帧采集时间
      </T>
      <T x={1410} y={905} anchor="middle" size={35}>
        一帧采集时间
      </T>
      {[
        { x: 210, w: 600, period: TRADE.slowPeriod },
        { x: 1240, w: 340, period: TRADE.fastPeriod },
      ].map((b, i) => (
        <g key={i}>
          <path d={`M${b.x} 947 h${b.w}`} stroke="#DDE6E9" strokeWidth="16" />
          <path
            d={`M${b.x} 947 h${b.w * clamp((local % b.period) / b.period)}`}
            stroke={C.cyan}
            strokeWidth="16"
          />
          <path d={`M${b.x + b.w} 930 v34`} stroke={C.muted} strokeWidth="3" />
          <circle
            cx={b.x + b.w}
            cy="947"
            r={5 + 5 * (1 - clamp((local % b.period) / 8))}
            fill={C.cyan}
          />
        </g>
      ))}
    </g>
  );
};
