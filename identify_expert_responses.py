"""
Pattern-matching candidate generator for EXPERT responses: given patient-query
candidates (output of identify_patient_queries.py), pulls the comment tree for
each post and flags comments that look like they come from a self-identified
mental-health professional — to pair as (client_statement, expert_response)
rows for human validation, the same shape as the existing dpo_* dataset.

Note: the post scraper (scrape_reddit.py) only pulled submissions, not
comments, so this script re-hits the Reddit API per post. Keep --limit modest
and respect the sleep between posts.

Usage:
    python identify_expert_responses.py data/candidate_patient_queries.csv \
        --tiers high medium --limit 150 --out data/candidate_expert_responses.csv
"""

import argparse
import csv
import re
import time

import praw

CLIENT_ID = "NxgQ31Kx6oLAaqSrbg-yhg"
CLIENT_SECRET = "wstfr23DOlO-jOq4T-qNTcaJEAIkbg"
USER_AGENT = "research"

# ───────────────────────── Expert-identity patterns ─────────────────────────

# First-person professional self-identification in the comment body. Requires
# "I am/I'm a ___" phrasing so we don't match a patient saying "my therapist
# is a ___" or "I saw a psychologist".
SELF_ID = re.compile(
    r"\bi(?:'m| am)\s+(?:a|an)\s+"
    r"(?:licensed\s+|certified\s+|practicing\s+|registered\s+|trainee\s+)?"
    r"(?:cbt\s+|clinical\s+|registered\s+)?"
    r"(therapist|psychologist|psychiatrist|counselor|counsellor|social\s+worker|"
    r"clinician|psychotherapist|lcsw|lmft|lpc|lpcc|lcpc|lmhc|psyd)\b",
    re.IGNORECASE,
)
SPEAKING_AS = re.compile(
    r"\b(?:speaking|writing)\s+as\s+a\s+(?:therapist|psychologist|clinician|counselor)\b"
    r"|\bas\s+a\s+(?:practicing\s+|licensed\s+)?(?:cbt\s+)?(?:therapist|psychologist|clinician|counselor|social\s+worker)\s*[,:]?\s*(?:i|this|in\s+my)\b"
    r"|\bin\s+my\s+(?:clinical\s+)?(?:practice|experience\s+as\s+a)\b"
    r"|\bsource:\s*(?:i'?m|i\s+am)?\s*(?:a\s+)?(?:therapist|psychologist|clinician|lcsw|lmft)\b",
    re.IGNORECASE,
)
CREDENTIAL_SIGNOFF = re.compile(
    r"\b(LCSW|LMFT|LPC|LPCC|LCPC|LMHC|PsyD|LCADC|RP|RSW)\b"
)

# Flair-based credential signal (many clinician subs require flair).
FLAIR_CREDENTIAL = re.compile(
    r"\b(therapist|psychologist|psychiatrist|counsel(?:l)?or|clinician|"
    r"social\s+worker|lcsw|lmft|lpc|lpcc|lcpc|lmhc|psyd|rp\b|rsw)\b",
    re.IGNORECASE,
)
# Flair text that disqualifies. These subs use "(Unverified)" as a suffix on
# a *claimed* credential (e.g. "Therapist (Unverified)" = self-identified,
# not mod-verified -- still a signal), so "unverified" alone must NOT
# disqualify. The actual non-pro labels are things like "NAT/Not a Therapist"
# or "Unverified: May Not Be a Therapist".
FLAIR_DISQUALIFY = re.compile(
    r"\bNAT\b|\bnot\s+a\s+therapist\b|\bmay\s+not\s+be\s+a\s+therapist\b|"
    r"\bstudent\b|\bintern\b|\btrainee\b|\bclient\b|\blayperson\b",
    re.IGNORECASE,
)


def classify_comment(body: str, flair: str):
    evidence = []
    flair = flair or ""
    body = body or ""

    flair_match = bool(FLAIR_CREDENTIAL.search(flair)) and not FLAIR_DISQUALIFY.search(flair)
    if flair_match:
        evidence.append(f"flair:'{flair.strip()}'")

    self_id = SELF_ID.search(body)
    if self_id:
        evidence.append(f"self_id:'{self_id.group(0)}'")

    speaking_as = SPEAKING_AS.search(body)
    if speaking_as:
        evidence.append(f"speaking_as:'{speaking_as.group(0)}'")

    signoff = CREDENTIAL_SIGNOFF.search(body)
    if signoff:
        evidence.append(f"signoff:'{signoff.group(0)}'")

    strong = bool(self_id or speaking_as or (flair_match and signoff))
    weak = bool(flair_match or signoff) and not strong

    if strong:
        tier = "high"
    elif weak:
        tier = "medium"
    else:
        tier = None

    return tier, "; ".join(evidence)


# Professional self-ID conventions (flair, "as a therapist..." replies) are
# concentrated in subs where pros are expected to weigh in. Symptom-support
# subs (Anxiety, OCD, depression, ...) mostly get peer replies with no
# credential signal at all, so sampling "top score across everything" mostly
# wastes API calls there. Prioritize these subs; --all-subs disables this.
PRO_RESPONSE_SUBS = {
    "askatherapist", "askpsychologists", "askpsychology", "TalkTherapy",
    "PsychotherapyHelp", "TherapistCouch", "ClinicalPsychology",
    "Psychologists", "therapy",
}


def fetch_candidates(in_path, tiers, limit, all_subs=False):
    with open(in_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["confidence_tier"] in tiers]
    if not all_subs:
        rows = [r for r in rows if r["Subreddit"] in PRO_RESPONSE_SUBS]
    rows.sort(key=lambda r: float(r["patient_query_score"]), reverse=True)
    return rows[:limit]


def collect(in_path, tiers, limit, out_path, max_comments_per_post=30, all_subs=False):
    posts = fetch_candidates(in_path, tiers, limit, all_subs)
    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
    print(f"Reddit connected (read-only: {reddit.read_only})")
    print(f"Scanning comments on {len(posts)} candidate patient-query posts...\n")

    out_rows = []
    for i, post_row in enumerate(posts, 1):
        post_id = post_row["Post_ID"]
        found_this_post = 0
        try:
            submission = reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:max_comments_per_post]:
                author = getattr(comment, "author", None)
                if author is None:
                    continue
                flair = comment.author_flair_text or ""
                tier, evidence = classify_comment(comment.body, flair)
                if tier is None:
                    continue
                out_rows.append({
                    "Post_ID": post_id,
                    "Subreddit": post_row["Subreddit"],
                    "patient_query_score": post_row["patient_query_score"],
                    "patient_query_tier": post_row["confidence_tier"],
                    "Patient_Title": post_row["Title"],
                    "Patient_Body": post_row["Body"],
                    "Comment_ID": comment.id,
                    "Comment_Author": str(author),
                    "Comment_Author_Flair": flair,
                    "Comment_Score": comment.score,
                    "Expert_Response": comment.body,
                    "expert_match_tier": tier,
                    "expert_match_evidence": evidence,
                    "is_valid_expert_response": "",
                    "annotation_notes": "",
                })
                found_this_post += 1
            print(f"[{i}/{len(posts)}] r/{post_row['Subreddit']} {post_id}: {found_this_post} expert-candidate comments")
        except Exception as e:
            print(f"[{i}/{len(posts)}] {post_id}: ERROR {e}")
        time.sleep(1.5)

    if out_rows:
        out_rows.sort(key=lambda r: (r["expert_match_tier"] != "high", -r["Comment_Score"]))
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        high = sum(1 for r in out_rows if r["expert_match_tier"] == "high")
        print(f"\nFound {len(out_rows)} candidate expert responses ({high} high-confidence) -> {out_path}")
    else:
        print("\nNo expert-candidate comments found in this sample.")

    return out_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="output of identify_patient_queries.py")
    p.add_argument("--tiers", nargs="+", default=["high", "medium"])
    p.add_argument("--limit", type=int, default=150, help="max patient-query posts to scan comments on")
    p.add_argument("--max-comments-per-post", type=int, default=30)
    p.add_argument("--all-subs", action="store_true",
                    help="scan score-ranked posts from every subreddit instead of just the ones where pros reliably reply")
    p.add_argument("--out", default="data/candidate_expert_responses.csv")
    args = p.parse_args()

    collect(args.input, set(args.tiers), args.limit, args.out, args.max_comments_per_post, args.all_subs)


if __name__ == "__main__":
    main()
