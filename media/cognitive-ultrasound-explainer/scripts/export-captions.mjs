import fs from "node:fs";
import path from "node:path";
const captions = JSON.parse(fs.readFileSync("src/captions.json", "utf8"));
const stamp = (ms) =>
  new Date(ms).toISOString().slice(11, 23).replace(".", ",");
fs.mkdirSync("docs", { recursive: true });
const srt = captions
  .map(
    (c, i) =>
      `${i + 1}\n${stamp(c.startMs)} --> ${stamp(c.endMs)}\n${c.text}\n`,
  )
  .join("\n");
fs.writeFileSync("docs/narration.srt", srt);
fs.writeFileSync(
  "docs/narration.md",
  "# 中文旁白稿\n\n115 秒，当前成片无音轨。以下文稿与烧录字幕一致，后续配音需对齐标注时间。\n\n" +
    captions
      .map((c) => `- **${c.startMs / 1000}—${c.endMs / 1000} 秒**：${c.text}`)
      .join("\n\n") +
    "\n",
);
const out = path.resolve("../../outputs/cognitive-ultrasound-explainer");
for (const file of ["narration.srt", "narration.md"])
  fs.copyFileSync("docs/" + file, path.join(out, file));
console.log("Exported narration and SRT");
