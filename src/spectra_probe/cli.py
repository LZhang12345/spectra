from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import bucket_mass, global_buckets, js_divergence_bits, ndcg_at_k
from .probing import (
    NO_VERBALIZERS,
    YES_VERBALIZERS,
    answer_token_ids,
    generative_classification,
    likelihood_probe,
    single_token_ids,
)
from .prompts import CHOICE_QUESTIONS, PROBE_QUESTIONS
from .taxonomies import MOVIELENS_GENRES
from .toy import toy_example


def run_demo() -> dict:
    example = toy_example()
    tail = global_buckets(example.popularity[None, :])["tail"]
    rows = []
    for name, prediction in (
        ("Direct generation, top 3", example.direct_generation(3)),
        ("Direct generation, top 5", example.direct_generation(5)),
        ("SPECTRA, likelihood probing", example.likelihood.distribution),
        ("SPECTRA, generative classification", example.generative.distribution),
    ):
        rows.append(
            {
                "method": name,
                "js_divergence_bits": example.js(prediction),
                "ndcg_at_10": ndcg_at_k(prediction, example.truth, 10),
                "tail_mass": bucket_mass(prediction, tail),
                "categories_with_mass": int(np.sum(np.asarray(prediction) > 0.0)),
            }
        )
    return {"true_tail_mass": bucket_mass(example.truth, tail), "rows": rows}


def print_demo() -> None:
    result = run_demo()
    print(f"{'Method':<38}{'JS (bits)':>10}{'NDCG@10':>9}{'Tail mass':>11}{'Categories':>12}")
    for row in result["rows"]:
        print(
            f"{row['method']:<38}{row['js_divergence_bits']:>10.3f}{row['ndcg_at_10']:>9.3f}"
            f"{row['tail_mass']:>11.3f}{row['categories_with_mass']:>9d}/19"
        )
    print(f"True tail mass: {result['true_tail_mass']:.3f}")


def run_probe(args: argparse.Namespace) -> None:
    from .model import HFNextTokenLogits

    categories = (
        [line.strip() for line in Path(args.categories).read_text().splitlines() if line.strip()]
        if args.categories
        else list(MOVIELENS_GENRES)
    )
    lm = HFNextTokenLogits(
        args.model,
        adapter_path=args.adapter,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        revision=args.revision,
    )
    yes_ids = single_token_ids(lm.tokenizer, YES_VERBALIZERS)
    no_ids = single_token_ids(lm.tokenizer, NO_VERBALIZERS)
    probe_question = PROBE_QUESTIONS[args.template][args.horizon]
    choice_question = CHOICE_QUESTIONS[args.template][args.horizon]
    outputs = []
    with open(args.records) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if args.variant == "likelihood":
                readout = likelihood_probe(
                    lm, record["context"], categories, probe_question, yes_ids, no_ids, args.temperature
                )
            else:
                readout = generative_classification(
                    lm,
                    record["context"],
                    categories,
                    choice_question,
                    answer_token_ids(lm.tokenizer, categories),
                    args.temperature,
                )
            row = {"id": record.get("id"), "distribution": readout.as_dict()}
            if "truth" in record:
                truth = [float(record["truth"].get(c, 0.0)) for c in categories]
                row["js_divergence_bits"] = js_divergence_bits(readout.distribution, truth)
                row["ndcg_at_10"] = ndcg_at_k(readout.distribution, truth, 10)
            outputs.append(row)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(outputs, indent=2))
    print(f"wrote {len(outputs)} readouts to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectra_probe")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    probe = sub.add_parser("probe")
    probe.add_argument("--model", required=True)
    probe.add_argument("--adapter", default=None)
    probe.add_argument("--records", required=True)
    probe.add_argument("--categories", default=None)
    probe.add_argument("--output", required=True)
    probe.add_argument("--variant", choices=["likelihood", "generative"], default="likelihood")
    probe.add_argument("--template", choices=sorted(PROBE_QUESTIONS), default="imperative")
    probe.add_argument("--horizon", choices=["long_term", "short_term"], default="long_term")
    probe.add_argument("--temperature", type=float, default=1.0)
    probe.add_argument("--device", default=None)
    probe.add_argument("--dtype", default="auto")
    probe.add_argument("--batch-size", type=int, default=8)
    probe.add_argument("--revision", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        print_demo()
    else:
        run_probe(args)
    return 0
