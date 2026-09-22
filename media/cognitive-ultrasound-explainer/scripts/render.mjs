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
  out = path.resolve(
    root,
    "../../outputs/cognitive-ultrasound-explainer/ppt-v2",
  );
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
const inputProps = {};
try {
  const composition = await selectComposition({
    serveUrl,
    id: "CognitiveUltrasoundPPT",
    inputProps,
    puppeteerInstance: browser,
  });
  if (process.argv.includes("--stills")) {
    const times = process.env.QA_TIMES
      ? process.env.QA_TIMES.split(",").map(Number)
      : [
          0, 1.2, 2.4, 4.4, 7, 10.5, 12.5, 14, 18, 23, 24.5, 27, 31, 35, 38.5,
          40.5, 42.7, 45.8, 47.3, 49, 52.5, 54,
        ];
    for (const t of times) {
      await renderStill({
        serveUrl,
        composition,
        inputProps,
        puppeteerInstance: browser,
        frame: Math.min(1649, Math.round(t * 30)),
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
    const outputLocation = path.join(out, "cognitive-ultrasound-ppt-v2.mp4");
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
          console.log(`Render ${pct}% (${p.renderedFrames}/1650)`);
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
