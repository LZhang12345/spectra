# Reproduction scope

## Included

- The SPECTRA readouts: likelihood probing, generative classification, and hierarchical probing
- The prompt templates and verbalizer sets for the MovieLens experiments
- The MovieLens 19-genre category list
- The evaluation metrics used in the paper
- A loader for Hugging Face causal language models with an optional LoRA adapter
- A synthetic demo, a tiny-model example, and tests that run on CPU

## Not included

- Fine-tuned adapters and the training pipeline
- The MovieLens and Yelp data. Download them from their maintainers under their own terms.
- The short-video platform data, which is proprietary. Only aggregate results appear in the paper.
- The full evaluation cohorts and job scripts behind the paper's tables

## Running the readout on your own data

1. Write each user's history as text. One JSON object per line, with a `context` field and an optional `truth` distribution.
2. Choose the categories. Pass a text file with one category per line through `--categories`.
3. Run `python -m spectra_probe probe` with your model and adapter.
4. Compare the output with a decoded list using `ranking_to_vec` and the metrics in `spectra_probe.metrics`.

The numbers in the paper come from models fine-tuned on each dataset. A model without that fine-tuning will give different results.

## Regenerating the figures

```bash
python -m pip install -e '.[plots]'
python scripts/build_readme_assets.py
python scripts/build_method_animation.py
```

Both scripts use the synthetic user in `spectra_probe.toy` and write their values next to the images in `assets/`.
