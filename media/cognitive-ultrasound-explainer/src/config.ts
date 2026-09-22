export const FPS = 30;
export const DURATION = 115;
export const C = {
  bg: "#08151e",
  text: "#edf4f5",
  muted: "#9cb0bb",
  line: "#28414e",
  cyan: "#68dce5",
  purple: "#b4a0ee",
  orange: "#f1ac65",
};
export const SCENES = [
  { start: 0, end: 22, title: "一条扫描线，从一次倾听开始" },
  { start: 22, end: 38, title: "从一条线，到跳动的心脏" },
  { start: 38, end: 62, title: "看得广、采得密、更新快" },
  { start: 62, end: 80, title: "不必把所有方向都测一遍？" },
  { start: 80, end: 105, title: "哪里拿不准，下一帧重点看哪里" },
  { start: 105, end: 115, title: "让采集与重建相互配合" },
];
export const LINE_SETS = [
  [-42, -25, -8, 9, 26, 43],
  [-36, -16, 8, 15, 22, 40],
  [-37, -23, -12, -2, 22, 39],
];
export const DENSE = Array.from({ length: 18 }, (_, i) => -45 + (i * 90) / 17);
export const SPARSE = Array.from({ length: 6 }, (_, i) => -45 + i * 18);
export const clamp = (x: number) => Math.max(0, Math.min(1, x));
export const smooth = (x: number) => {
  const p = clamp(x);
  return p * p * (3 - 2 * p);
};
export const phase = (f: number) => ((Math.floor(f) % 48) + 48) % 48;
