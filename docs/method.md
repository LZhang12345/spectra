# Method notes

## Setting

A user has an interaction history $X$. The categories $C=\{c_1,\dots,c_K\}$ are named and interpretable, such as movie genres. The target $\theta_*$ is the user's empirical distribution over $C$ in a later period.

## Alignment

The LLM is first adapted to the domain. Continued pretraining on text built from user and item metadata and interaction traces teaches the domain vocabulary. LoRA fine-tuning then trains the model to predict the user's category distribution from the history. The paper uses Qwen3-8B, and repeats the experiments with DeepSeek-R1-Distill-Llama-8B.

This repository does not include training code or adapters. The readout works with any causal language model that returns next-token logits.

## Readout

SPECTRA builds one score per category and applies a softmax, $\theta_S=\operatorname{softmax}(S/\tau)$.

**Likelihood probing.** Each category gets its own prompt:

```text
{context}

Question: {question about one category}

Answer: 
```

The score is the mean logit of the affirmative tokens minus the mean logit of the negative tokens. `single_token_ids` keeps only the verbalizers that the tokenizer encodes as one token, with and without a leading space. The default sets are `Yes, Y, y` and `No, N, n`.

**Generative classification.** One prompt lists the categories as lettered options and asks for a letter. The score of a category is the logit of its letter. This needs one forward pass per user and supports up to 26 categories per prompt.

**Hierarchical probing.** For a two-level taxonomy, `hierarchical_probe` reads a distribution over coarse categories and then a conditional distribution inside each selected branch. The probability of a fine category is the product of the two. Branches that are not expanded keep their coarse probability, which is returned separately as `unexpanded`. The cost falls from one prompt per fine category to one prompt per coarse category plus one per fine category in the selected branches.

## Direct generation baseline

Direct generation asks the model to output a ranked list of $k$ categories. `ranking_to_vec` turns a list into a distribution by spreading probability evenly over the listed categories. Every category outside the list gets zero probability, whatever the model's internal scores were.

## Theory in the paper

The paper compares the two approaches on ranking-quality metrics.

- If SPECTRA's scores are calibrated well enough relative to the gap between the user's $k$-th and $(k+1)$-th category, SPECTRA recovers the user's true top-$k$ categories.
- Under frequency-biased decoding, a decoded list misses tail categories with positive probability, and any distribution built only from that list cannot assign them mass.

The synthetic demo illustrates the second point. Its decoded list is ranked by the user's true preference multiplied by global popularity.

## Metrics

| Function | Meaning |
| :--- | :--- |
| `js_divergence_bits` | Jensen-Shannon divergence in bits, between 0 and 1 |
| `ndcg_at_k` | NDCG with the true probability of each category as its relevance |
| `entropy_bits` | Shannon entropy of one distribution |
| `topk_exposure_entropy` | Entropy of how often each category appears in users' top-$k$ lists |
| `global_buckets` | Head, middle, and tail thirds of the categories by average true mass |
| `bucket_mass` | Probability that a prediction places on a bucket |
