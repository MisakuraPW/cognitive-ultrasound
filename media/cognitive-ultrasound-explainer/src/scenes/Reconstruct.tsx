import React from "react";
import { C, smooth } from "../config";
import { T, Label, Arrow } from "../visuals";
export const Reconstruct: React.FC<{ t: number }> = ({ t }) => (
  <g opacity={smooth(t - 62)}>
    {t < 68 ? (
      <g>
        <T x={1140} y={375} size={49} weight={600}>
          时间有限
        </T>
        <T x={1140} y={448} size={49} weight={600}>
          先看哪里？
        </T>
        <T x={1140} y={572} size={33} color={C.muted}>
          从分散的少量方向开始
        </T>
      </g>
    ) : (
      <g>
        <Label x={1120} y={360}>
          实测：青蓝实线
        </Label>
        <T x={1177} y={420} size={30} color={C.muted}>
          线外区域，暂时没有测量
        </T>
        {t >= 73 && (
          <g opacity={smooth((t - 73) / 1.5)}>
            <Label x={1120} y={515} color={C.purple} dashed>
              估计：紫色虚线边界
            </Label>
            <T x={1177} y={575} size={30} color={C.muted}>
              已测信息 + 学到的图像规律
            </T>
            <Arrow x={1177} y={636} length={325} color={C.purple} />
            <T x={1177} y={714} size={40} color={C.purple}>
              形成完整的重建估计
            </T>
            <T x={1177} y={777} size={28} color={C.muted}>
              补充的细节，可能有误
            </T>
          </g>
        )}
      </g>
    )}
  </g>
);
