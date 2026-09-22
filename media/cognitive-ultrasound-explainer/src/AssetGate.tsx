import { useEffect, useState } from "react";
import {
  delayRender,
  continueRender,
  cancelRender,
  staticFile,
} from "remotion";
// SVG images are predecoded so a frame can never capture an unloaded texture.
export const AssetGate = () => {
  const [handle] = useState(() =>
    delayRender("Decode deterministic phantom textures"),
  );
  useEffect(() => {
    const names = ["observed", "estimate"].flatMap((kind) =>
      Array.from(
        { length: 48 },
        (_, i) => `${kind}-${String(i).padStart(2, "0")}.png`,
      ),
    );
    names.push("possible-0.png", "possible-1.png", "possible-2.png");
    Promise.all(
      names.map(
        (name) =>
          new Promise<void>((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
              img.decode().then(() => resolve(), reject);
            };
            img.onerror = () => reject(new Error(name));
            img.src = staticFile("phantom/" + name);
          }),
      ),
    )
      .then(() => continueRender(handle))
      .catch(cancelRender);
  }, [handle]);
  return null;
};
