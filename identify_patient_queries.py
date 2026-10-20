"""
Pattern-matching candidate generator: flags Reddit posts that look like a
patient/client describing a personal problem and seeking help (as opposed to
professional shop-talk, career/training questions, surveys, or pure
psychoeducation) — mirroring the "client_statement" role in CBT-Bench /
CBT-PsychoPref style datasets.

This is a precision-oriented first pass, NOT a classifier: it produces a
scored + tiered CSV that a human annotator reviews and corrects (is_valid
column left blank for annotation), the same workflow already used in
data/annotations.csv. Tune the weights/thresholds once you have annotated a
sample and can measure precision/recall per tier.

Usage:
    python identify_patient_queries.py data/reddit_posts_*.csv
    python identify_patient_queries.py data/reddit_posts_*.csv --out data/candidate_patient_queries.csv
"""

import argparse
import csv
import glob
import re
import sys


# ───────────────────────── Subreddit priors ─────────────────────────
# Not exclusionary — just a weight. Clinician subs still contain genuine
# self-disclosure (e.g. a psychiatrist posting about their own anxiety);
# patient subs still contain pure psychoeducation / announcement posts.

CLINICIAN_SUBS = {
    "therapists", "psychotherapists", "TherapistsInTherapy", "PsychotherapyHelp",
    "TherapistCouch", "Psychologists", "ClinicalPsychology", "askpsychologists",
}
MIXED_SUBS = {
    "CBT", "CBTpractice", "CBTselfhelp", "CBTI", "REBT", "CognitiveDefusion",
    "MBCT", "dbtselfhelp", "DBT", "schematherapy", "Mindfulness", "therapy",
    "TalkTherapy", "askatherapist", "askpsychology", "psychology",
}
# Everything else (Anxiety, OCD, depression, CPTSD, ptsd, insomnia,
# mentalhealth, HealthAnxiety, PanicDisorder, socialanxiety, BDD,
# traumatoolbox, selfhelp, DecidingToBeBetter, AmIOverthinking, ...) is
# treated as patient-facing by default.

SUBREDDIT_WEIGHT = {"clinician": -1.0, "mixed": 0.0, "patient": 1.5}


def subreddit_bucket(sub: str) -> str:
    if sub in CLINICIAN_SUBS:
        return "clinician"
    if sub in MIXED_SUBS:
        return "mixed"
    return "patient"


# ───────────────────────── Pattern libraries ─────────────────────────

def _compile(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# Help-seeking framing — the "ask" of a client statement.
HELP_SEEKING = _compile([
    r"\bany\s*(?:advice|tips|suggestions|thoughts)\b",
    r"\bdoes\s+anyone\s+(?:else\s+)?(?:know|have|deal|struggle|experience|feel)\b",
    r"\bhas\s+anyone\s+(?:else\s+)?(?:dealt|experienced|been|had|struggled)\b",
    r"\bhow\s+do\s+(?:i|you)\s+(?:cope|deal|stop|get\s+over|handle|manage)\b",
    r"\bwhat\s+should\s+i\s+do\b",
    r"\bi\s+don'?t\s+know\s+what\s+to\s+do\b",
    r"\bi\s+don'?t\s+know\s+how\s+to\b",
    r"\b(?:need|needing|looking\s+for)\s+(?:some\s+)?(?:advice|help|support|guidance)\b",
    r"\bhelp\s+needed\b",
    r"\bcan\s+(?:anyone|someone)\s+(?:help|relate|explain)\b",
    r"\bis\s+(?:this|it)\s+normal\b",
    r"\bam\s+i\s+(?:overreacting|overthinking|crazy|alone|the\s+only\s+one)\b",
    r"\bhow\s+can\s+i\s+(?:stop|get|fix|overcome|improve)\b",
    r"\bwhat\s+(?:helped|works?|worked)\s+(?:you|for\s+you)\b",
])

# First-person distress / symptom language. Deliberately broad-spectrum
# (anxiety, depression, OCD, trauma, panic, insomnia, general dysregulation)
# rather than per-diagnosis, since the subreddit already narrows topic.
SYMPTOM_LANGUAGE = _compile([
    r"\bi('|’)?m\s+(?:so\s+|really\s+|constantly\s+)?(?:anxious|depressed|panicking|spiraling|struggling|overwhelmed|exhausted|terrified|scared|hopeless|worthless|numb)\b",
    r"\bi\s+(?:feel|felt|keep\s+feeling)\s+(?:like\s+)?(?:a\s+failure|worthless|hopeless|numb|anxious|depressed|overwhelmed|out\s+of\s+control|trapped|stuck)\b",
    r"\bi\s+can'?t\s+(?:stop|sleep|focus|cope|breathe|get\s+out\s+of\s+bed|stop\s+thinking|stop\s+crying|eat)\b",
    r"\bintrusive\s+thoughts?\b",
    r"\b(?:panic|anxiety)\s+attacks?\b",
    r"\bracing\s+thoughts?\b",
    r"\b(?:compulsions?|compulsive|obsessions?|obsessing)\b",
    r"\bflashbacks?\b",
    r"\btriggered?\b",
    r"\brumina(?:te|ting|tion)\b",
    r"\bself[- ]harm\b",
    r"\bsuicidal\b",
    r"\bi\s+(?:keep|can'?t\s+stop)\s+(?:crying|shaking|spiraling)\b",
    r"\bi\s+relapsed\b",
    r"\bmy\s+anxiety\b",
    r"\bmy\s+depression\b",
    r"\bmy\s+ocd\b",
    r"\bmy\s+ptsd\b",
    r"\bmy\s+therapist\s+(?:says|told|wants)\b",   # patient referencing OWN therapist
    r"\bi\s+was\s+diagnosed\s+with\b",
    r"\bi\s+have\s+(?:been\s+)?(?:diagnosed\s+with\s+)?(?:anxiety|depression|ocd|ptsd|cptsd|bpd|adhd|bdd)\b",
])

# Professional/meta discussion — the author is speaking AS a clinician,
# trainee, or researcher, not as someone seeking help for their own problem.
CLINICIAN_IDENTITY = _compile([
    r"\bmy\s+clients?\b",
    r"\bmy\s+patients?\b",
    r"\bmy\s+caseload\b",
    r"\bas\s+a\s+(?:cbt\s+)?(?:therapist|psychologist|psychiatrist|clinician|counselor|counsellor|social\s+worker)\b",
    r"\bi(?:'|’)?m\s+a\s+(?:licensed\s+)?(?:cbt\s+)?(?:therapist|psychologist|psychiatrist|clinician|counselor|counsellor|lcsw|lmft|lpc|lpcc|psyd)\b",
    r"\bin\s+my\s+practice\b",
    r"\bin\s+session\s+with\s+a\s+client\b",
    r"\bsupervisee\b|\bsupervisor\b",
    r"\bintern\s+hours\b|\bpracticum\b",
    r"\bcontinuing\s+education\b|\bCEUs?\b",
])

META_DISCUSSION = _compile([
    r"\blicensure\b|\baccreditation\b",
    r"\bgrad(?:uate)?\s+(?:school|program)\b",
    r"\bletter\s+of\s+recommendation\b",
    r"\bwhich\s+(?:program|degree|school)\s+should\b",
    r"\breferring?\s+out\b",
    r"\bprivate\s+practice\b.*\b(?:billing|insurance|caseload|marketing)\b",
    r"\bbachelor|master'?s?\s+program\b",
    r"\bPWP\b",  # psychological wellbeing practitioner role discussion (UK IAPT)
])

QUESTION_MARK = re.compile(r"\?")
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.IGNORECASE)
WORD = re.compile(r"\b\w+\b")


def score_post(title: str, body: str, subreddit: str):
    text = f"{title}\n{body}".strip()
    words = WORD.findall(text)
    n_words = max(len(words), 1)

    fp_count = len(FIRST_PERSON.findall(text))
    fp_ratio = fp_count / n_words

    helpseeking_hits = [p.pattern for p in HELP_SEEKING if p.search(text)]
    symptom_hits = [p.pattern for p in SYMPTOM_LANGUAGE if p.search(text)]
    clinician_hits = [p.pattern for p in CLINICIAN_IDENTITY if p.search(text)]
    meta_hits = [p.pattern for p in META_DISCUSSION if p.search(text)]
    has_question = bool(QUESTION_MARK.search(text))

    bucket = subreddit_bucket(subreddit)
    sub_weight = SUBREDDIT_WEIGHT[bucket]

    score = (
        1.0 * has_question
        + 1.0 * min(fp_ratio * 20, 2.0)          # cap contribution
        + 1.5 * min(len(helpseeking_hits), 3)
        + 1.5 * min(len(symptom_hits), 4)
        + sub_weight
        - 3.0 * len(clinician_hits)
        - 2.0 * len(meta_hits)
    )

    if clinician_hits and len(clinician_hits) >= 1 and bucket == "clinician":
        # Clear professional-voice post in a clinician subreddit -> exclude outright.
        tier = "exclude"
    elif score >= 4.5:
        tier = "high"
    elif score >= 2.0:
        tier = "medium"
    elif score >= 0.5:
        tier = "low"
    else:
        tier = "exclude"

    return {
        "patient_query_score": round(score, 2),
        "confidence_tier": tier,
        "subreddit_bucket": bucket,
        "has_question": has_question,
        "first_person_ratio": round(fp_ratio, 3),
        "helpseeking_matches": len(helpseeking_hits),
        "symptom_matches": len(symptom_hits),
        "clinician_identity_matches": len(clinician_hits),
        "meta_discussion_matches": len(meta_hits),
        "matched_patterns": "; ".join(
            [f"help:{p[:30]}" for p in helpseeking_hits]
            + [f"sym:{p[:30]}" for p in symptom_hits]
            + [f"clin:{p[:30]}" for p in clinician_hits]
            + [f"meta:{p[:30]}" for p in meta_hits]
        ),
    }


def process(in_files, out_path):
    rows_out = []
    for path in in_files:
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                features = score_post(row.get("Title", ""), row.get("Body", ""), row.get("Subreddit", ""))
                rows_out.append({**row, **features, "is_valid_patient_query": "", "annotation_notes": ""})

    rows_out.sort(key=lambda r: r["patient_query_score"], reverse=True)

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    tiers = {}
    for r in rows_out:
        tiers[r["confidence_tier"]] = tiers.get(r["confidence_tier"], 0) + 1
    print(f"Scored {len(rows_out)} posts -> {out_path}")
    for t in ("high", "medium", "low", "exclude"):
        print(f"  {t:8s}: {tiers.get(t, 0)}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="scraped post CSV(s), globs ok")
    p.add_argument("--out", default="data/candidate_patient_queries.csv")
    args = p.parse_args()

    files = []
    for pattern in args.inputs:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        print("No input files matched.", file=sys.stderr)
        sys.exit(1)

    process(files, args.out)


if __name__ == "__main__":
    main()
