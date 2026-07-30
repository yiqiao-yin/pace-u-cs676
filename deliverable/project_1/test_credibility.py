"""
test_credibility.py — contract tests for score_url().

    python test_credibility.py

These check the SHAPE of your output, not its quality. They must keep passing
however much you rewrite the internals — the app, the grader, and evaluate.py
all rely on this contract. Use `evaluate.py` to measure quality.

Deliverable 1 asks for "initial testing to validate input/output handling".
This file is that, and adding your own cases here is part of the deliverable.

No pytest required, deliberately — one less thing to install.
"""

from credibility import score_band, score_url

PASSED = 0
FAILED = 0


def check(condition: bool, description: str) -> None:
    """Assert-with-a-label so one failure doesn't stop the whole run."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {description}")
    else:
        FAILED += 1
        print(f"  FAIL  {description}")


print("\nContract: return shape")
result = score_url("https://www.nature.com/articles/example", use_llm=False)
check(isinstance(result, dict), "returns a dict")
check(set(result.keys()) == {"score", "explanation"}, "has exactly the keys 'score' and 'explanation'")
check(isinstance(result["score"], float), "score is a float")
check(isinstance(result["explanation"], str), "explanation is a str")
check(len(result["explanation"]) > 0, "explanation is not empty")

print("\nContract: score range")
for url in [
    "https://www.nature.com/x",
    "https://medium.com/@a/b",
    "http://unknown-site.xyz/page",
    "https://en.wikipedia.org/wiki/X",
]:
    score = score_url(url, use_llm=False)["score"]
    check(0.0 <= score <= 1.0, f"{url[:40]:<42} -> {score:.2f} is within [0, 1]")

print("\nContract: malformed input is handled, not raised")
for bad in ["", "   ", "not a url", "ftp://files.example.com/x", "javascript:alert(1)", "//example.com"]:
    try:
        bad_result = score_url(bad, use_llm=False)
        ok = isinstance(bad_result, dict) and 0.0 <= bad_result["score"] <= 1.0
        check(ok, f"{bad!r:<28} -> {bad_result['score']:.2f} (no exception)")
    except Exception as e:
        check(False, f"{bad!r:<28} raised {type(e).__name__}")

print("\nContract: determinism")
a = score_url("https://arxiv.org/abs/1706.03762", use_llm=False)
b = score_url("https://arxiv.org/abs/1706.03762", use_llm=False)
check(a == b, "same URL scored twice gives the same result")

print("\nSanity: ordering the baseline should already get right")
journal = score_url("https://www.nature.com/articles/x", use_llm=False)["score"]
blog = score_url("https://randomblog.blogspot.com/x", use_llm=False)["score"]
check(journal > blog, f"a journal ({journal:.2f}) outranks a personal blog ({blog:.2f})")

gov = score_url("https://www.census.gov/data", use_llm=False)["score"]
throwaway = score_url("http://whatever.xyz/page", use_llm=False)["score"]
check(gov > throwaway, f"a .gov source ({gov:.2f}) outranks a throwaway domain ({throwaway:.2f})")

print("\nContract: score_band")
check(score_band(0.9)[0] == "HIGH", "0.90 -> HIGH")
check(score_band(0.5)[0] == "MEDIUM", "0.50 -> MEDIUM")
check(score_band(0.1)[0] == "LOW", "0.10 -> LOW")

print(f"\n{'=' * 60}")
print(f"  {PASSED} passed, {FAILED} failed")
print(f"{'=' * 60}\n")
raise SystemExit(1 if FAILED else 0)
