# Playful Title Filter (+ Optional Title Generation)

Detect playful/funny paper titles from CSPaperSum, and optionally fine-tune a small LLM to generate titles from abstracts.

## Data
- Source: CSPaperSum — https://arxiv.org/abs/2502.20582
- The notebook builds a clean `data/titles_abstracts.csv` on run.
- To keep the repo lightweight, a small preview file `data/titles_abstracts_sample.csv` may be included; the full CSV is regenerated locally by the notebook.

## Method (Filtering)
Two signals:
1) **Heuristic keywords/patterns** → `rule_score`
2) **Zero-shot MNLI** (`facebook/bart-large-mnli` on GPU or a smaller distil MNLI on CPU) → `zs_playful_prob`

The heuristic is mapped to [0,1] with a logistic squash (center ≈ 1.5) and averaged with the model probability:

```
ensemble_score = 0.5 * rule_norm + 0.5 * zs_playful_prob
playful_flag   = ensemble_score ≥ 0.55
```

## Results
- The notebook prints the fraction of titles flagged as playful and writes:
  - `results/playful_titles.csv`
  - `results/neutral_title_examples.csv`
- (Optional SFT) It also saves `results/sft_titles_validation_sample.csv` with generated titles and ROUGE-L scores.

## Additional Title Generation (Optional)
- Fine-tunes **FLAN-T5-small** on abstract → title pairs and evaluates with ROUGE-L.

## Run
Install deps: `pip install -r requirements.txt`

Then open the single notebook in `notebooks/` and **Run All** (or run sections in order):
1. Build `data/titles_abstracts.csv`
2. Run the playful-title filter → writes `results/*.csv`
3. (Optional) Fine-tune FLAN-T5-small → writes `results/sft_titles_validation_sample.csv`

## Notes
- Raise thresholds (`ZS_THRESH`, `ENSEMBLE_THRESH`) for precision; lower for recall.
- Cascade the zero-shot model only on mid-range `rule_score` titles for speed.
- Large checkpoints are intentionally not committed; the full dataset is regenerated locally.
