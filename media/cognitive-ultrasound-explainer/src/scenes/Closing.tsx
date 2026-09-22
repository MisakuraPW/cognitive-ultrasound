import React from "react";
import { C, smooth } from "../config";
import { T, Label } from "../visuals";
export const Closing: React.FC<{ t: number }> = ({ t }) => (
  <g opacity={smooth(t - 105)}>
    <T x={1100} y={354} size={43} color={C.muted}>
      有限的采集时间
    </T>
    <T x={1100} y={436} size={55} weight={600}>
      优先观察
    </T>
    <T x={1100} y={510} size={55} weight={600}>
      更值得看的位置
    </T>
    <path d="M1100 560 H1760" stroke={C.line} strokeWidth="2" />
    <Label x={1100} y={633}>
      采集
    </Label>
    <Label x={1350} y={633} color={C.purple} dashed>
      重建估计
    </Label>
    <T x={1100} y={740} size={28} color={C.muted}>
      速度收益还取决于重建计算与设备实现
    </T>
    <T x={1100} y={790} size={26} color={C.muted}>
      研究原理介绍 · 非临床效果或实时设备验证
    </T>
  </g>
);
