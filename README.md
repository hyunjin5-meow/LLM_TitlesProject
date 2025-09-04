# Playful Title Filter (+ Optional Title Generation)

**Goal.** Detect playful/funny paper titles from CSPaperSum, and fine-tune a small LLM to generate titles from abstracts.

## Data
- Source: CSPaperSum (https://arxiv.org/abs/2502.20582)
- Built `data/titles_abstracts.csv` in `notebooks/step1_data.ipynb`.

## Method (Filtering)
Two signals:
1) Heuristic keywords/patterns → `rule_score`
2) Zero-shot MNLI (`bart-large-mnli` or distil MNLI) → `zs_playful_prob`

We normalize the heuristic (logistic around 1.5) and average with the model prob:
`ensemble_score = 0.5*rule_norm + 0.5*zs_playful_prob`, then flag with threshold `0.55`.

## Results
- Flagged playful: <X>% of <N> titles.
- Artifacts: `results/playful_titles.csv`, `results/neutral_title_examples.csv`.

## Additional Title Generation
- FLAN-T5-small fine-tuned on abstract→title.
- Examples & ROUGE-L: `results/sft_titles_validation_sample.csv`.

## Run
