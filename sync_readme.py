"""Sync README's Results section with the latest evaluation outputs.

Usage:
    python sync_readme.py

Reads:
    evaluation/evaluation_summary.csv
    evaluation/dataset_size.txt

Writes:
    README.md (rewrites the section between the README-RESULTS markers)
"""
from pathlib import Path

import pandas as pd

SUMMARY = Path("evaluation/evaluation_summary.csv")
SIZE_FILE = Path("evaluation/dataset_size.txt")
README = Path("README.md")

START = "<!-- README-RESULTS:START -->"
END   = "<!-- README-RESULTS:END -->"

if not SUMMARY.exists() or not SIZE_FILE.exists():
    raise SystemExit("Run python run_evaluation.py first.")

summary = pd.read_csv(SUMMARY)
dataset_size = int(SIZE_FILE.read_text().strip())

base = summary.set_index("Metric")["Baseline"].to_dict()
prop = summary.set_index("Metric")["Proposed"].to_dict()

section = f"""## 18. Results

Evaluated on **{dataset_size:,} movies** after cleaning. Test set: 12 well-known
movies spanning multiple genres and decades. Relevance is defined as **sharing
at least one genre** with the source movie (a qualitative proxy, not user
satisfaction).

### Baseline vs Proposed

| Metric | Baseline | Proposed |
|---|---|---|
| Precision@5 | {base['Precision@5']:.3f} | {prop['Precision@5']:.3f} |
| Precision@10 | {base['Precision@10']:.3f} | {prop['Precision@10']:.3f} |
| Recall@10 | {base['Recall@10']:.3f} | {prop['Recall@10']:.3f} |
| Catalog coverage (200 queries) | {base['Coverage (200 queries)']*100:.1f}% | {prop['Coverage (200 queries)']*100:.1f}% |
| Mean genre diversity | {base['Mean genre diversity']:.2f} | {prop['Mean genre diversity']:.2f} |
| Query time (ms) | {base['Query time (ms)']:.1f} | {prop['Query time (ms)']:.1f} |

### Interpretation

- **Precision@5 and Precision@10 improved.** The proposed model reorders
  the top of the list using rating, popularity, and preference signals,
  which lifts genre-overlap precision relative to pure similarity ranking.
- **Recall@10 improved slightly.** Recall is bounded above by roughly 0.33
  under top-50 pooling; values near 0.20 represent about 60% of the
  achievable ceiling.
- **Catalog coverage decreased marginally.** The ranking layer concentrates
  on high-quality candidates, so slightly fewer distinct movies appear
  across 200 queries. The change is small (a few percent).
- **Mean genre diversity decreased slightly.** Diversity prioritises
  variety within each top-10 list, but combined with the ranking bias
  toward highly-rated titles, the average genre spread per list drops
  marginally.
- **Query time increased.** The proposed model runs similarity, ranking,
  normalization, and diversity passes, so it is measurably slower than the
  baseline. Both remain interactive for a Streamlit GUI.

> **These are ranking metrics, not accuracy.** Precision@K measures content
> relevance, not user satisfaction. Not every metric improves for every
> query — the Proposed model trades a small amount of coverage, diversity,
> and speed for higher top-of-list precision.
"""

text = README.read_text(encoding="utf-8")

if START in text and END in text:
    before = text.split(START)[0]
    after = text.split(END)[1]
    new = before + START + "\n" + section + END + after
else:
    new = text + "\n\n" + START + "\n" + section + END + "\n"

README.write_text(new, encoding="utf-8")
print(f"✓ README updated with {dataset_size} movies and current metrics.")