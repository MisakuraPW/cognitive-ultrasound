import React from "react";
import type { Caption } from "@remotion/captions";
import captions from "./captions.json";
import { C } from "./config";
export const Captions: React.FC<{ frame: number }> = ({ frame }) => {
  const c = (captions as Caption[]).find(
    (c) => (frame * 1000) / 30 >= c.startMs && (frame * 1000) / 30 < c.endMs,
  );
  return (
    <div
      style={{
        position: "absolute",
        left: 100,
        right: 100,
        top: 925,
        height: 90,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 39,
        lineHeight: 1.5,
        color: C.text,
        textAlign: "center",
        fontWeight: 500,
      }}
    >
      {c?.text}
    </div>
  );
};
