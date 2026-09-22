import { Composition } from "remotion";
import { CognitiveUltrasound } from "./Composition";
import "./index.css";
export const RemotionRoot = () => (
  <Composition
    id="CognitiveUltrasoundPPT"
    component={CognitiveUltrasound}
    durationInFrames={1650}
    fps={30}
    width={1920}
    height={1080}
  />
);
