/** One word of the utterance, carrying both surfaces plus the slot karaoke will fill in. */
export interface Token {
  /** Orthographic word as the user typed it. */
  text: string;
  /** Its canonical IPA, from vernacula-phonemizer. */
  ipa: string;
  /** Playback window in seconds of the final audio — estimated, see inference/alignment.ts. */
  start?: number;
  end?: number;
}

export interface Utterance {
  lang: string;
  text: string;
  ipa: string;
  tokens: Token[];
  /** 24 kHz mono. */
  audio: Float32Array;
  sampleRate: number;
}

export type Stage = "idle" | "loading-models" | "phonemizing" | "generating" | "ready" | "error";

export interface Progress {
  stage: Stage;
  /** 0..1 within the current stage, or undefined when indeterminate. */
  fraction?: number;
  detail?: string;
}
