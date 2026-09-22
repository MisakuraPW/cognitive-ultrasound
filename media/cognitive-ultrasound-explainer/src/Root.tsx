import { Composition } from "remotion";
import { CognitiveUltrasound } from "./Composition";
import "./index.css";
export const RemotionRoot = () => (
  <Composition
    id="CognitiveUltrasound"
    component={CognitiveUltrasound}
    durationInFrames={3450}
    fps={30}
    width={1920}
    height={1080}
    defaultProps={{ voiceover: "" }}
  />
);
