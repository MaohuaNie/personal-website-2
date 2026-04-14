"""
Test the MailerLite email broadcast without running the full digest pipeline.

Loads the most recent entry from digests.json and calls send_digest_email()
with that data — no LLM calls, no paper search, no API cost beyond MailerLite.

Triggered manually via the "Test Email Broadcast" GitHub Actions workflow,
or locally with:

    MAILERLITE_API_TOKEN=... python3 test_email.py

Exits non-zero if the broadcast fails, so CI surfaces the error clearly.
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def main():
    if not os.getenv("MAILERLITE_API_TOKEN"):
        print("ERROR: MAILERLITE_API_TOKEN env var is not set.")
        sys.exit(1)

    # Import after env check so the no-op branch in send_digest_email doesn't hide issues
    from paper_search import send_digest_email, DIGEST_INDEX

    if not DIGEST_INDEX.exists():
        print(f"ERROR: {DIGEST_INDEX} does not exist.")
        sys.exit(1)

    with open(DIGEST_INDEX, "r", encoding="utf-8") as f:
        index = json.load(f)
    if not index:
        print("ERROR: digests.json is empty.")
        sys.exit(1)

    latest = sorted(index, key=lambda x: x["date"], reverse=True)[0]
    date_str = latest["date"]
    filename = latest["file"]
    paper_count = int(latest.get("papers", 0))
    title = latest.get("title", "")

    # Parse "Research Digest · 2026-03-16 → 2026-03-31" into (start_day, end_day)
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*→\s*(\d{4}-\d{2}-\d{2})", title)
    if m:
        start_day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        end_day = datetime.strptime(m.group(2), "%Y-%m-%d").date()
    else:
        # Fallback — use the digest date for both ends
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        start_day, end_day = d, d

    print(f"Sending test email using latest digest:")
    print(f"  date:     {date_str}")
    print(f"  file:     {filename}")
    print(f"  papers:   {paper_count}")
    print(f"  range:    {start_day} → {end_day}")
    print()

    # send_digest_email only uses len(results) for the paper count, so a list
    # of any N placeholders is enough to reproduce a real send.
    fake_results = [None] * paper_count

    send_digest_email(date_str, filename, fake_results, start_day, end_day)
    print()
    print("Done. Check your inbox in ~1 minute.")


if __name__ == "__main__":
    main()
