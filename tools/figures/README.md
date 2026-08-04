# Figure generators

Course figures drawn from scratch with numpy and matplotlib, so the repository owns
them outright and they can be restyled or regenerated at any time.

| Script | Produces | Used by |
| --- | --- | --- |
| `make_metrics_figures.py` | `pics/12_metrics_01.png`, `pics/12_metrics_02.png`, `pics/12_metrics_03.png` | [Classification Metrics](../../docs/12_classification_metrics.md) |

```bash
pip install numpy matplotlib
python tools/figures/make_metrics_figures.py
```

Run it from the repository root; paths are resolved relative to the script, so the
output always lands in `pics/`.

## Conventions

- **Draw, do not extract.** The slide deck contains third-party material, so figures
  for the notes are generated rather than lifted from it.
- **Name as `NN_topic_MM.png`**, matching the rest of `pics/` — the number is the
  session the figure belongs to.
- **White background** (`facecolor="white"`). The site is dark, and its CSS gives
  every image a white backing plate, so figures should assume a light background.
- **No emoji.** matplotlib's default font (DejaVu Sans) has no emoji glyphs, so
  they render as empty boxes. Draw the shape instead.
- **Deterministic.** Seed any randomness, so rerunning produces the same image and
  git does not see a diff for no reason.

Students are welcome to copy any of this for their own project write-ups.
