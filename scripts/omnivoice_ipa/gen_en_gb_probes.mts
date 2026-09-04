import { writeFileSync } from "node:fs";
import { phonemizeAsync } from "/home/chris/Programming/vernacula-phonemizer/src/index.ts";
const PROBES: [string, string, string][] = [
  ["smoke_reported",  "White smoke rose from the plant.",            "əᶷ — the reported 'smoke'->'smik' failure"],
  ["show_reported",   "The show goes on and on.",                    "əᶷ — the reported 'show'->'shoe' failure"],
  ["televisions",     "Televisions and radios were stolen.",         "the reported 'televisi'epots'"],
  ["goat_dense",      "There is no snow on the road home.",          "əᶷ x4"],
  ["square",          "He dared to share the fare with her.",        "ɛə"],
  ["cure",            "The tourist was sure of the cure.",           "ʊə"],
  ["fire",            "The fire in the choir grew higher.",          "aᶦə"],
  ["hour",            "Our flower tower is over there.",             "aᶷə"],
  ["choice",          "The boy enjoyed the noise of the toy.",       "ɔᶦ"],
  ["near",            "We hear the deer is near here.",              "ɪə"],
];
const out: Record<string, {text:string; ipa_gb:string; ipa_us:string; targets:string}> = {};
for (const [k, text, targets] of PROBES) {
  out[k] = { text, targets,
    ipa_gb: (await phonemizeAsync(text, "en-GB")).replace(/\s+/gu, " ").trim(),
    ipa_us: (await phonemizeAsync(text, "en")).replace(/\s+/gu, " ").trim() };
}
writeFileSync("/mnt/data/omnivoice_ipa/train/en_gb_probes.json", JSON.stringify(out, null, 1) + "\n");
for (const [k, v] of Object.entries(out)) console.log(`${k}\n  GB ${v.ipa_gb}\n  US ${v.ipa_us}`);
