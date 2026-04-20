#!/usr/bin/env python3
"""
Template-based synthetic drug dialogue for the en-medical corpus.

Addresses the chemical-F1 gap we observed when we removed written
medical text (DailyMed, PubMed) from the corpus: spoken-register
sources (MTSamples + synthetic dialogue) don't mention specialty drug
names often enough to give the LM strong priors on them. Adding
written DailyMed back biased the decoder toward formal register at a
fluency cost.

This script generates ~5-10M words of *speech-register* drug dialogue
by filling clinical-speech templates with (drug, condition, dose, ...)
tuples drawn from a gazetteer extracted from DailyMed. Each sentence
reads like something a clinician or patient would actually say:

    "I've been taking amoxicillin for my sinusitis."
    "Patient is on atorvastatin 20 mg and metformin 500 mg."
    "Did the olanzapine help with your symptoms?"
    "I'd like to switch from lansoprazole to omeprazole."

Output: one sentence per line, cased + punctuated, ready to concatenate
into the en-medical layered corpus.
"""
import argparse
import gzip
import json
import random
import sys
from pathlib import Path


# Templates grouped by speaker/register. Each uses {drug}, {drug2}, {drug3}
# for multi-med mentions; {cond} for condition; {dose}/{freq}/{dur} for
# scheduling; {sym} for side-effect symptoms.

DOCTOR_DICTATION = [
    "Patient was started on {drug}.",
    "Patient was started on {drug} {dose} for {cond}.",
    "Patient is currently on {drug} {dose} {freq}.",
    "We'll initiate {drug} at {dose} {freq}.",
    "I've started her on {drug} for her {cond}.",
    "I've started him on {drug} for {cond}.",
    "Continue {drug} at current dose.",
    "Increase {drug} to {dose}.",
    "Decrease {drug} to {dose}.",
    "Discontinue {drug}.",
    "Switch from {drug} to {drug2}.",
    "Add {drug} {dose} {freq}.",
    "Patient is allergic to {drug}.",
    "Medications include {drug}, {drug2}, and {drug3}.",
    "Current medications: {drug}, {drug2}.",
    "Plan: start {drug} {dose} {freq} for {cond}.",
    "Refill {drug} {dose}.",
    "Patient reports good tolerance to {drug}.",
    "Patient reports {sym} on {drug}, will consider dose reduction.",
    "Follow up in {dur} to reassess {drug} therapy.",
    "History of {cond}, on {drug}.",
    "Chronic {cond}, managed with {drug}.",
    "Acute {cond}, treated with {drug}.",
]

DOCTOR_CONSULT = [
    "I'd like to start you on {drug} for your {cond}.",
    "I'm going to prescribe {drug} for that.",
    "We'll try you on {drug} and see how you do.",
    "Take {drug} {dose} {freq}.",
    "Have you been taking the {drug}?",
    "How's the {drug} working for you?",
    "Are you having any side effects from the {drug}?",
    "Let's increase your {drug} to {dose}.",
    "Let's decrease your {drug} to {dose}.",
    "I want you to stop taking the {drug}.",
    "We're going to switch you from {drug} to {drug2}.",
    "Any issues with the {drug}?",
    "Have you been taking the {drug} as prescribed?",
    "How long have you been on the {drug}?",
    "Do you remember how much {drug} you take?",
    "Are you currently taking any medications?",
    "Have you ever had a reaction to {drug}?",
    "The {drug} should help with your {cond}.",
    "You may experience some {sym} when starting {drug}.",
    "The {drug} can cause {sym} in some people.",
]

PATIENT = [
    "I've been taking {drug} for my {cond}.",
    "The doctor put me on {drug}.",
    "I take {drug} every day.",
    "I'm on {drug} for my {cond}.",
    "I take {drug} {dose} {freq}.",
    "I stopped taking {drug} because it made me feel {sym}.",
    "I ran out of my {drug} last week.",
    "Can I get a refill of my {drug}?",
    "My {cond} got worse after I stopped {drug}.",
    "I'm on {drug} and {drug2}.",
    "The {drug} has been giving me {sym}.",
    "I've been on {drug} for about {dur} now.",
    "I forgot to take my {drug} this morning.",
    "Is it okay to take {drug} with {drug2}?",
    "Does {drug} have any side effects?",
    "I think the {drug} is helping.",
    "I don't think the {drug} is working.",
    "I'm allergic to {drug}.",
    "My {cond} has been well controlled on {drug}.",
    "I've been feeling {sym} since I started {drug}.",
    "How long will I need to take the {drug}?",
    "Can you write me a prescription for {drug}?",
]

EXCHANGES = [  # doctor + patient back-to-back
    ("Are you on any medications?", "Yes, I take {drug} for my {cond}."),
    ("Are you on any medications?", "I'm on {drug} and {drug2}."),
    ("What are you currently taking?", "Just {drug} {dose} {freq}."),
    ("How long have you been on {drug}?", "About {dur} now."),
    ("Any side effects from the {drug}?", "I get {sym} sometimes."),
    ("Any side effects from the {drug}?", "No, I feel fine on it."),
    ("Do you remember the dose?", "I think it's {dose} {freq}."),
    ("Did the {drug} help?", "Yeah, my {cond} is much better."),
    ("Did the {drug} help?", "Not really, I'm still having {sym}."),
    ("Have you ever had a reaction to {drug}?", "Yeah, it gave me {sym}."),
    ("Are you allergic to anything?", "I can't take {drug}, it gives me {sym}."),
    ("When did you start {drug}?", "About {dur} ago."),
    ("Who prescribed the {drug}?", "My regular doctor did."),
    ("Is anything else going on?", "Just my {cond}, which is why I'm on {drug}."),
]

DOSES = [
    "5 mg", "10 mg", "20 mg", "25 mg", "40 mg", "50 mg", "75 mg", "100 mg",
    "150 mg", "200 mg", "250 mg", "500 mg", "750 mg", "1000 mg",
    "one tablet", "two tablets", "half a tablet", "one capsule",
    "5 milligrams", "ten milligrams", "twenty milligrams",
    "1 mg", "2 mg", "2.5 mg", "0.5 mg",
    "10 milliliters", "5 ml", "one teaspoon", "5 ccs",
]

FREQUENCIES = [
    "once a day", "twice a day", "three times a day", "four times a day",
    "every morning", "every evening", "at bedtime", "with meals",
    "as needed", "every six hours", "every eight hours", "every twelve hours",
    "daily", "BID", "TID", "QID", "PRN", "in the morning", "at night",
    "every other day", "three times a week", "weekly",
]

DURATIONS = [
    "a few days", "a week", "two weeks", "a month", "two months",
    "three months", "six months", "about a year", "over a year",
    "two years", "several years", "a long time", "since my diagnosis",
]

SIDE_EFFECTS = [
    "nausea", "headaches", "dizziness", "fatigue", "dry mouth",
    "drowsiness", "stomach upset", "diarrhea", "constipation", "insomnia",
    "a rash", "some itching", "muscle pain", "weight gain", "weight loss",
    "tremors", "heart palpitations", "shortness of breath", "blurred vision",
    "sleepiness", "anxiety", "feeling jittery", "cold hands",
]


def load_gazetteer(path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text())
    return data["drugs"], data["diseases"]


def fill_template(template: str, drugs: list[str], conds: list[str]) -> str:
    def pick_drug():  return random.choice(drugs)
    def pick_cond():  return random.choice(conds)
    result = template
    # Use unique drug picks for {drug}, {drug2}, {drug3}
    picks = random.sample(drugs, min(3, len(drugs)))
    replacements = {
        "{drug}":  picks[0],
        "{drug2}": picks[1] if len(picks) > 1 else pick_drug(),
        "{drug3}": picks[2] if len(picks) > 2 else pick_drug(),
        "{cond}":  pick_cond(),
        "{dose}":  random.choice(DOSES),
        "{freq}":  random.choice(FREQUENCIES),
        "{dur}":   random.choice(DURATIONS),
        "{sym}":   random.choice(SIDE_EFFECTS),
    }
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result


def generate(args) -> int:
    drugs, conds = load_gazetteer(args.gazetteer)
    print(f"gazetteer: {len(drugs)} drugs, {len(conds)} diseases", file=sys.stderr)

    rng = random.Random(args.seed)
    random.seed(args.seed)

    open_ctx = gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) \
               if args.output.suffix == ".gz" else args.output.open("w", encoding="utf-8")

    pools = [
        (DOCTOR_DICTATION, 0.30),
        (DOCTOR_CONSULT,   0.30),
        (PATIENT,          0.30),
    ]
    # Exchanges yield two sentences each, weight half as often
    exch_weight = 0.10

    words = 0
    lines = 0
    with open_ctx as fout:
        while words < args.target_words:
            roll = rng.random()
            cumulative = 0.0
            emitted: list[str] = []
            for pool, w in pools:
                cumulative += w
                if roll < cumulative:
                    emitted.append(fill_template(rng.choice(pool), drugs, conds))
                    break
            else:
                # Fell through — emit an exchange (both sides)
                q, a = rng.choice(EXCHANGES)
                emitted.append(fill_template(q, drugs, conds))
                emitted.append(fill_template(a, drugs, conds))
            for s in emitted:
                fout.write(s)
                fout.write("\n")
                words += s.count(" ") + 1
                lines += 1
            if lines % 200_000 == 0:
                print(f"  {lines:,} lines, {words:,} words",
                      file=sys.stderr)

    print(f"\n[total] {lines:,} lines, {words:,} words -> {args.output}",
          file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gazetteer",    required=True, type=Path,
                    help="JSON file with {'drugs': [...], 'diseases': [...]}")
    ap.add_argument("--output",       required=True, type=Path)
    ap.add_argument("--target-words", type=int, default=8_000_000)
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
