# Project page

A buildless GitHub Pages site. Preview from the repository root:

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory pages
```

Open http://127.0.0.1:8765. The existing `.github/workflows/deploy-pages.yml` publishes this directory when its files are pushed to `main`.

- `index.html`: hero, the one-line thesis with an Actor → Judge → Teacher summary (the Judge card shows JEV's milestone / other / no-op score heatmap), a tabbed region with one chapter per role, main takeaways, and code.
- `style.css`: warm paper palette, role colors, chapter backgrounds, interactive charts, and the dark takeaways section.
- `script.js`: scroll progress, reveal-on-scroll, nav highlighting, Actor / Judge / Teacher chapter tabs (left/right arrows, sticky tab bar, "Next" buttons, `#actor`/`#judge`/`#teacher` deep links), and the interactive charts (goal masking toggle, context slider, Value/Progress toggle, Teacher gain baseline toggle).
- `assets/`: standalone page assets, including exported paper figures.

## Narrative

The page follows one thread: **tasks with dependency** (ALFWorld Clean, Heat, Cool).

1. **Actor** fails on them (JEV 14.7%). Goal masking shows the model knows the required step (14.6% → 77.6%); a dependency hint does not fix it.
2. **Judge** can credit those steps given the right **context** (Hindsight: "open microwave" 0.20 → 0.76) and **question** (Progress: milestone − no-op gap 0.05 → 0.36).
3. **Teacher** uses the same context and question; the 1.5B policy reaches 86.8% on these tasks. Against the strongest of the other three settings, Hindsight + Progress wins on every task type JEV fails as an Actor and loses on Pick and Pick2, which JEV handles well.

The Main Takeaways section restates `sections/04_understanding_decision_models.tex` (Takeaways) as five cards, each paired with one number from the paper.

## Content and figure sources

Content was checked against the paper at `/Users/dk/Desktop/refine` on 2026-11-13: authors from `main.tex`, abstract from `sections/01_abstract.tex`, findings and takeaways from `sections/04_understanding_decision_models.tex`, Actor results from `tables/actor_performance.tex`, policy results from `tables/policy_results.tex`.

Derived numbers:

- Dependency-task success per Teacher setting (75.0 / 79.9 / 70.6 / 86.8) is the mean of Clean, Heat, and Cool weighted by evaluation task count (27, 16, 25, from the Actor table percentages). It reproduces the paper's 86.8% and its "6.9 to 16.2" margin.
- "Gain vs. strongest other query" is Hindsight + Progress minus the best of the other three 1.5B settings, per task type. "Gain vs. GRPO" is Hindsight + Progress minus GRPO.
- The context-slider scores are the values in `figures/evaluator-context/context_case_heatmap.pdf`.

PNG assets are raster exports of these paper PDFs (overview at 2×; analysis figures at 1800 px wide):

| Page asset            | Paper source under `figures/`               |
| --------------------- | ------------------------------------------- |
| overview.png          | overview/figure1_vector.pdf                 |
| actor-performance.png | actor-performance/actor_performance.pdf     |
| actor-comparison.png  | actor-comparison/actor_comparison.pdf       |
| judge-objective.png   | evaluator-objective/evaluator_objective.pdf |
| judge-context.png     | evaluator-context/context_case_heatmap.pdf  |

`banner.jpg` is a JPEG export of the repository's `assets/banner-cropped.png`. The overview removes 118 pt from the right edge, matching the trim in the paper's `overview.tex`. No paper source files were modified.

Role colors come from the overview: Actor `#c39448`, Judge `#4d6bf3` (also the paper's finding-box color), Teacher `#709b6a`, ink `#151b3b`. The page uses system fonts and has no external runtime dependencies. Without JavaScript all three chapters are shown in sequence, and the charts show their initial state. Motion respects `prefers-reduced-motion`. Paper/model/data buttons and BibTeX are omitted until public URLs and citation metadata are confirmed.
