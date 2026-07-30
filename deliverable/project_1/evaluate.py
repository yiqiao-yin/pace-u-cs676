"""
evaluate.py — measure how good your scorer is.

    python evaluate.py            # rules only (fast, free, no API key needed)
    python evaluate.py --llm      # include the Claude judgment layer

Prints a per-URL table and three summary numbers. Run it before you change
anything so you have a baseline, then run it after each change. Deliverable 2
asks you to justify your approach — "MAE went from 0.28 to 0.14" is a much
better justification than "it seems better now".

THE LABELS BELOW ARE A STARTING POINT, NOT GROUND TRUTH. They were assigned by
one person in one sitting. Part of the exercise is arguing with them: if you
think a label is wrong, say so in your report and defend it. Extending this set
with sources from your own domain is encouraged and is worth credit.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

from credibility import score_band, score_url

# (url, expected_score, why_this_label)
LABELLED_URLS: List[Tuple[str, float, str]] = [
    # --- High: peer reviewed, archival, or strong editorial standards --------
    ("https://www.nature.com/articles/s41586-021-03819-2", 0.95, "peer-reviewed journal article"),
    ("https://www.nejm.org/doi/full/10.1056/NEJMoa2034577", 0.95, "peer-reviewed medical journal"),
    ("https://pubmed.ncbi.nlm.nih.gov/33301246/", 0.90, "indexed biomedical literature"),
    ("https://www.reuters.com/world/example-report-2024-01-01/", 0.85, "wire service, corrections policy"),
    ("https://www.census.gov/data/tables/2023/demo/income-poverty.html", 0.90, "primary government statistics"),
    ("https://apnews.com/article/example-story-12345", 0.85, "wire service"),

    # --- Medium: useful but needs care --------------------------------------
    ("https://arxiv.org/abs/1706.03762", 0.65, "influential BUT a preprint, not peer reviewed"),
    ("https://en.wikipedia.org/wiki/Statistical_learning_theory", 0.60, "well-sourced but tertiary and open-edit"),
    ("https://www.biorxiv.org/content/10.1101/2020.01.01.000001v1", 0.50, "preprint, no review"),
    ("https://scikit-learn.org/stable/modules/linear_model.html", 0.75, "authoritative docs for its own library"),

    # --- Low: self-published, user-generated, or promotional ----------------
    ("https://medium.com/@someone/why-ai-is-magic-abc123", 0.25, "self-published, no editorial review"),
    ("https://randomblog.blogspot.com/2024/03/my-thoughts.html", 0.15, "personal blog"),
    ("https://www.reddit.com/r/MachineLearning/comments/abc123/", 0.20, "user-generated comment thread"),
    ("https://example.com/sponsored/miracle-supplement", 0.10, "sponsored commercial content"),

    # --- Near zero: satire and fabrication -----------------------------------
    ("https://www.theonion.com/study-finds-example-1849", 0.05, "satire, factually false by design"),
    ("http://totally-legit-news.xyz/shocking-truth", 0.05, "no provenance, insecure, throwaway TLD"),

    # --- HELD OUT: domains deliberately absent from DOMAIN_SCORES ------------
    # The block above is easy for the baseline because those domains are in its
    # lookup table. These are not. They are where a table-driven approach falls
    # apart and where a real algorithm earns its keep — expect most of your
    # remaining error to live down here.
    ("https://www.pnas.org/doi/10.1073/pnas.2020123118", 0.92, "peer-reviewed academy journal"),
    ("https://jamanetwork.com/journals/jama/fullarticle/2762130", 0.93, "peer-reviewed medical journal"),
    ("https://www.who.int/news-room/fact-sheets/detail/example", 0.88, "international health authority"),
    ("https://www.propublica.org/article/example-investigation", 0.85, "investigative newsroom, fact-checked"),
    ("https://www.imf.org/en/Publications/WEO/example", 0.85, "primary economic data publisher"),
    ("https://stackoverflow.com/questions/12345/how-to-do-x", 0.45, "often correct, but unreviewed and unattributed"),
    ("https://seekingalpha.com/article/example-stock-analysis", 0.35, "contributor-submitted, light editorial review"),
    ("https://health-truth-daily.info/miracle-cure-doctors-hate", 0.05, "fabricated health claims, no provenance"),
]


def evaluate(use_llm: bool) -> Dict[str, float]:
    """Score every labelled URL and report error against the expected values."""
    rows = []
    abs_errors: List[float] = []
    band_hits = 0

    for url, expected, rationale in LABELLED_URLS:
        result = score_url(url, use_llm=use_llm if use_llm else False)
        got = result["score"]
        error = abs(got - expected)
        abs_errors.append(error)

        # Band accuracy is often what matters in the UI: a reader mostly needs
        # the chip to be the right colour, not the decimal to be exact.
        if score_band(got)[0] == score_band(expected)[0]:
            band_hits += 1

        rows.append((url, expected, got, error, rationale))

    print(f"\n{'expected':>9} {'got':>6} {'err':>6}  url")
    print("-" * 100)
    for url, expected, got, error, rationale in sorted(rows, key=lambda r: -r[3]):
        flag = "  <-- worst" if error == max(abs_errors) else ""
        display = url if len(url) <= 58 else url[:55] + "..."
        print(f"{expected:>9.2f} {got:>6.2f} {error:>6.2f}  {display}{flag}")
        print(f"{'':>23}  ({rationale})")

    mae = sum(abs_errors) / len(abs_errors)
    band_accuracy = band_hits / len(LABELLED_URLS)
    worst = max(abs_errors)

    print("-" * 100)
    print(f"  URLs evaluated     : {len(LABELLED_URLS)}")
    print(f"  Mean absolute error: {mae:.3f}   (lower is better; 0.000 is perfect)")
    print(f"  Band accuracy      : {band_accuracy:.1%}   (HIGH/MEDIUM/LOW chip correct)")
    print(f"  Worst single error : {worst:.3f}")
    print(f"  LLM layer          : {'on' if use_llm else 'off (rules only)'}\n")

    return {"mae": mae, "band_accuracy": band_accuracy, "worst": worst}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate credibility.score_url against labelled URLs.")
    parser.add_argument("--llm", action="store_true", help="include the Claude judgment layer (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()
    evaluate(use_llm=args.llm)
