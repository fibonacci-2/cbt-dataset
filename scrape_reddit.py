"""
Minimal Reddit post scraper for the subreddit list in reddit.py.

Fetches LIMIT posts from each subreddit and writes them to one CSV.

Setup:
    pip install praw
Usage:
    python scrape_reddit.py                 # 100 newest posts per subreddit
    python scrape_reddit.py --sort top --limit 200
"""

import argparse
import csv
import os
import re
import time
from datetime import datetime, timezone

import praw

from reddit import subreddits as SUBREDDIT_URLS

CLIENT_ID = "NxgQ31Kx6oLAaqSrbg-yhg"
CLIENT_SECRET = "wstfr23DOlO-jOq4T-qNTcaJEAIkbg"
USER_AGENT = "research"

LIMIT = 100
OUTPUT_FOLDER = "data"

FIELDS = [
    "Post_ID", "Author", "Subreddit", "Created_UTC", "Year_Month",
    "Score", "Num_Comments", "Post_URL", "Title", "Body",
]


def subreddit_names(urls):
    """https://www.reddit.com/r/CBT/ -> CBT"""
    names = []
    for url in urls:
        m = re.search(r"/r/([^/]+)", url)
        if m and m.group(1) not in names:
            names.append(m.group(1))
    return names


def listing(subreddit, sort, limit):
    if sort == "hot":
        return subreddit.hot(limit=limit)
    if sort == "top":
        return subreddit.top(limit=limit, time_filter="all")
    if sort == "rising":
        return subreddit.rising(limit=limit)
    return subreddit.new(limit=limit)


def collect(sort="new", limit=LIMIT, output_folder=OUTPUT_FOLDER):
    names = subreddit_names(SUBREDDIT_URLS)
    os.makedirs(output_folder, exist_ok=True)

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )
    print(f"Reddit connected (read-only: {reddit.read_only})")
    print(f"{len(names)} subreddits | sort={sort} | limit={limit}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_folder, f"reddit_posts_{timestamp}.csv")
    seen_ids = set()
    total = 0

    with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for i, name in enumerate(names, 1):
            count = 0
            try:
                for post in listing(reddit.subreddit(name), sort, limit):
                    if post.id in seen_ids:
                        continue
                    seen_ids.add(post.id)

                    created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    writer.writerow({
                        "Post_ID": post.id,
                        "Author": str(post.author) if post.author else "[deleted]",
                        "Subreddit": name,
                        "Created_UTC": created,
                        "Year_Month": created.strftime("%Y-%m"),
                        "Score": post.score,
                        "Num_Comments": post.num_comments,
                        "Post_URL": f"https://www.reddit.com{post.permalink}",
                        "Title": post.title or "",
                        "Body": post.selftext if isinstance(post.selftext, str) else "",
                    })
                    count += 1
                f.flush()
                print(f"[{i}/{len(names)}] r/{name}: {count} posts")
            except Exception as e:
                print(f"[{i}/{len(names)}] r/{name}: ERROR {e}")
            total += count
            time.sleep(2)

    print(f"\nSaved {total} posts -> {out_file}")
    return out_file


def main():
    p = argparse.ArgumentParser(description="Scrape posts from the subreddits in reddit.py")
    p.add_argument("--sort", default="new", choices=["new", "hot", "top", "rising"])
    p.add_argument("--limit", type=int, default=LIMIT, help="posts per subreddit")
    p.add_argument("--output", default=OUTPUT_FOLDER)
    args = p.parse_args()
    collect(sort=args.sort, limit=args.limit, output_folder=args.output)


if __name__ == "__main__":
    main()
