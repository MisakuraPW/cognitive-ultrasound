export const FPS = 30;
export const DURATION = 55;
export const C = {
  bg: "#FFFFFF",
  text: "#30343B",
  muted: "#64717C",
  line: "#CBD5DA",
  cyan: "#007F98",
  cyanLight: "#58D6E5",
  purple: "#7451B8",
  orange: "#C87516",
  heat: "#F2A448",
};
export const SCENES = [
  { start: 0, end: 12, title: "连续扫描成像" },
  { start: 12, end: 24, title: "成像取舍" },
  { start: 24, end: 37, title: "当前帧重建" },
  { start: 37, end: 55, title: "不确定性与下一帧" },
];
export const MAIN = { x: 960, y: 225, scale: 1.27 };
export const LINE_SETS = [
  [-42, -25, -8, 9, 26, 43],
  [-36, -16, 8, 15, 22, 40],
  [-37, -23, -12, -2, 22, 39],
];
export const DENSE = Array.from({ length: 24 }, (_, i) => -46 + i * 4);
export const SPARSE = Array.from({ length: 9 }, (_, i) => -30 + i * 7.5);
export const TRADE = {
  start: 13,
  slowPeriod: 84,
  fastPeriod: 25,
  deepRadius: 555,
  shallowRadius: 435,
};
export const clamp = (v: number) => Math.max(0, Math.min(1, v));
export const smooth = (v: number) => {
  const p = clamp(v);
  return p * p * (3 - 2 * p);
};
export const phase = (f: number) => ((Math.floor(f) % 48) + 48) % 48;
