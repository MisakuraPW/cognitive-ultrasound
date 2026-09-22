import { bundle } from "@remotion/bundler";
import {
  selectComposition,
  renderStill,
  renderMedia,
  openBrowser,
  getVideoMetadata,
} from "@remotion/renderer";
import path from "node:path";
import fs from "node:fs";
const root = process.cwd(),
  out = path.resolve(root, "../../outputs/cognitive-ultrasound-explainer");
fs.mkdirSync(path.join(out, "previews"), { recursive: true });
const browserExecutable =
  process.env.REMOTION_BROWSER ||
  (process.platform === "win32"
    ? "C:/Program Files/Google/Chrome/Application/chrome.exe"
    : undefined);
const serveUrl = await bundle({
  entryPoint: path.join(root, "src/index.ts"),
  outDir: path.join(out, "bundle"),
  rspack: true,
});
const browser = await openBrowser("chrome", { browserExecutable });
const inputProps = { voiceover: process.env.VOICEOVER || "" };
try {
  const composition = await selectComposition({
    serveUrl,
    id: "CognitiveUltrasound",
    inputProps,
    puppeteerInstance: browser,
  });
  if (process.argv.includes("--stills")) {
    const times = process.env.QA_TIMES
      ? process.env.QA_TIMES.split(",").map(Number)
      : [
          0, 7, 9, 12, 18, 24, 28, 33, 40, 44, 49, 54, 59, 65, 70, 76, 83, 88,
          94, 98, 102, 108, 113,
        ];
    for (const t of times) {
      await renderStill({
        serveUrl,
        composition,
        inputProps,
        puppeteerInstance: browser,
        frame: Math.min(3449, Math.round(t * 30)),
        output: path.join(
          out,
          "previews",
          `frame-${String(t).padStart(5, "0")}.png`,
        ),
        imageFormat: "png",
      });
      console.log("Checked still", t);
    }
  } else {
    let last = -1;
    const outputLocation = path.join(out, "cognitive-ultrasound-explainer.mp4");
    await renderMedia({
      serveUrl,
      composition,
      inputProps,
      puppeteerInstance: browser,
      outputLocation,
      codec: "h264",
      crf: 18,
      pixelFormat: "yuv420p",
      imageFormat: "jpeg",
      jpegQuality: 95,
      concurrency: 3,
      onProgress: (p) => {
        const pct = Math.floor(p.progress * 100);
        if (pct >= last + 5) {
          last = pct;
          console.log(`Render ${pct}% (${p.renderedFrames}/3450)`);
        }
      },
    });
    const metadata = await getVideoMetadata(outputLocation);
    fs.writeFileSync(
      path.join(out, "video-metadata.json"),
      JSON.stringify(metadata, null, 2),
    );
    console.log(JSON.stringify(metadata));
  }
} finally {
  await browser.close({ silent: true });
}
