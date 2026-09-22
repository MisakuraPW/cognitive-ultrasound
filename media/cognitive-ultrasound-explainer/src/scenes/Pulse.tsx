import React from "react";
import { C, smooth } from "../config";
import { T, point, sector, Arrow } from "../visuals";
export const Pulse: React.FC<{ t: number }> = ({ t }) => {
  const travel = (t - 5) * 112,
    depths = [180, 310, 455],
    a = 12;
  return (
    <g>
      <g opacity={1 - smooth((t - 19) / 3)}>
        <path
          d="M245 262 Q425 248 600 266 M643 267 Q835 248 1000 263"
          stroke="#597580"
          strokeWidth="5"
          fill="none"
        />
        <T x={255} y={237} color={C.muted} size={27}>
          胸壁
        </T>
        <T x={682} y={210} size={28}>
          心超探头
        </T>
        <T x={260} y={845} size={25} color={C.muted}>
          简化心脏切面 · 传播过程慢放示意
        </T>
      </g>
      {t >= 4 && t < 16 && (
        <g transform="translate(620 270)">
          <path d={sector(a - 3, a + 3)} fill={C.cyan} opacity=".055" />
          {depths.map((d, i) => {
            const p = point(a, d);
            return (
              <g key={d}>
                <ellipse
                  cx={p[0]}
                  cy={p[1]}
                  rx="27"
                  ry="5"
                  fill="#b5c6cd"
                  opacity=".8"
                />
                <T x={p[0] + 43} y={p[1] + 8} size={27} color={C.muted}>
                  {["浅", "中", "深"][i]}
                </T>
              </g>
            );
          })}
          {travel > 0 &&
            travel < 550 &&
            [0, 10, 20].map((off) => {
              const p = point(a, travel - off);
              return (
                <path
                  key={off}
                  d={`M${p[0] - 22} ${p[1]} Q${p[0]} ${p[1] + 8} ${p[0] + 22} ${p[1] - 5}`}
                  stroke={C.cyan}
                  strokeWidth="3"
                  fill="none"
                />
              );
            })}
          {depths.map((d, i) => {
            const r = 2 * d - travel;
            if (travel < d || r < 8) return null;
            const p = point(a, r);
            return (
              <path
                key={d}
                d={`M${p[0] - 19} ${p[1]} Q${p[0]} ${p[1] - 9} ${p[0] + 19} ${p[1] + 3}`}
                stroke="#dbe6e9"
                strokeWidth={3 - i * 0.5}
                fill="none"
              />
            );
          })}
        </g>
      )}
      {t < 16 ? (
        <g opacity={smooth((t - 1) / 1.2)}>
          <T x={1170} y={330} size={48} weight={600}>
            发出短脉冲
          </T>
          <T x={1170} y={397} size={48} weight={600}>
            倾听不同深度的回声
          </T>
          <T x={1170} y={488} size={28} color={C.muted}>
            接收时间
          </T>
          <path d="M1170 574 H1770" stroke={C.line} strokeWidth="3" />
          {depths.map((d, i) => {
            const px = 1200 + d;
            return (
              <g key={d} opacity={t >= 5 + (2 * d) / 112 ? 1 : 0.22}>
                <path
                  d={`M${px - 10} 574 l5 -15 l5 39 l5 -${[85, 66, 48][i]} l5 61 l5 -24`}
                  stroke={C.cyan}
                  strokeWidth="3"
                  fill="none"
                />
                <T x={px} y={634} anchor="middle" color={C.muted}>
                  {["浅", "中", "深"][i]}
                </T>
              </g>
            );
          })}
          <T x={1170} y={720} size={39} color={C.cyan}>
            回来得晚 → 位置更深
          </T>
        </g>
      ) : (
        <g opacity={smooth((t - 16) / 1)}>
          <T x={1160} y={342} size={48} weight={600}>
            回声 → 亮暗
          </T>
          <T x={1160} y={410} size={30} color={C.muted}>
            接收的信号
          </T>
          <path
            d="M1160 475 h45 l8 -35 l10 68 l10 -82 l10 70 l10 -21 h65 l10 -23 l10 43 l10 -59 l10 52 l10 -13 h50"
            stroke={C.cyan}
            strokeWidth="3"
            fill="none"
          />
          <Arrow x={1312} y={540} length={90} />
          <rect
            x="1160"
            y="585"
            width="410"
            height="85"
            rx="12"
            fill="#132b36"
            stroke={C.line}
          />
          <T x={1365} y={640} anchor="middle" size={37}>
            信号处理
          </T>
          <T x={1160} y={748} size={35} color={C.cyan}>
            形成这一方向的扫描线
          </T>
        </g>
      )}
    </g>
  );
};
