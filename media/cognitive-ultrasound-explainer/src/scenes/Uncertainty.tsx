import React from "react";
import { C, smooth, LINE_SETS } from "../config";
import { Fan, T, Label, Arrow, Line } from "../visuals";
export const Uncertainty: React.FC<{ t: number }> = ({ t }) => {
  const stage = t < 97 ? 0 : t < 101 ? 1 : 2;
  return (
    <g opacity={smooth(t - 80)}>
      {t < 92 ? (
        <g>
          <T x={1140} y={320} size={40} weight={600}>
            几种可能的重建
          </T>
          {[0, 1, 2].map((i) => (
            <g key={i}>
              <Fan
                id={`possible-${i}`}
                x={1220 + i * 220}
                y={378}
                scale={0.26}
                mode="possible"
                variant={i}
                probe={false}
              />
              <ellipse
                cx={1246 + i * 220}
                cy={471}
                rx="23"
                ry="26"
                fill="none"
                stroke={C.orange}
                strokeWidth="2"
              />
              <T
                x={1220 + i * 220}
                y={577}
                anchor="middle"
                size={27}
                color={C.purple}
              >
                可能 {i + 1}
              </T>
            </g>
          ))}
          <T x={1140} y={652} size={35} color={C.orange}>
            局部意见不一致 → 更拿不准
          </T>
          <T x={1140} y={720} size={29} color={C.muted}>
            橙色不是病灶，也不是诊断概率
          </T>
          <T x={1140} y={772} size={27} color={C.muted}>
            不代表系统知道真实误差
          </T>
        </g>
      ) : (
        <g>
          <T x={1140} y={335} size={42} weight={600}>
            {stage === 0
              ? "把方向安排给下一帧"
              : stage === 1
                ? "下一帧：新的观察位置"
                : "再下一帧：继续更新"}
          </T>
          <Label x={1140} y={424}>
            实线：本帧已测
          </Label>
          <Label x={1140} y={483} color={C.orange} dashed>
            虚线：下一帧候选方向
          </Label>
          <T x={1140} y={580} size={42} color={C.cyan}>
            每帧 6 条示意线
          </T>
          <T x={1140} y={640} size={32} color={C.muted}>
            数量相同，位置改变
          </T>
          <T x={1140} y={727} size={29} color={C.muted}>
            更新之后，仍可能有拿不准的地方
          </T>
          {t < 104 && (
            <g
              transform="translate(620 270)"
              opacity={smooth(
                (t - (stage === 0 ? 92 : stage === 1 ? 98 : 102)) / 1,
              )}
            >
              {LINE_SETS[Math.min(stage + 1, 2)].map((a) => (
                <Line
                  key={a}
                  angle={a}
                  color={C.orange}
                  dash
                  opacity={0.55}
                  width={2}
                />
              ))}
            </g>
          )}
        </g>
      )}
      <g opacity={smooth((t - 90) / 1)}>
        <rect x="235" y="830" width="1450" height="62" rx="12" fill="#102631" />
        <T
          x={505}
          y={871}
          anchor="middle"
          size={30}
          color={stage === 0 ? C.text : C.muted}
        >
          这一帧：测量与重建
        </T>
        <Arrow x={778} y={858} length={210} color={C.orange} />
        <T
          x={1320}
          y={871}
          anchor="middle"
          size={30}
          color={stage > 0 ? C.cyan : C.orange}
        >
          下一帧：调整观察位置
        </T>
      </g>
    </g>
  );
};
