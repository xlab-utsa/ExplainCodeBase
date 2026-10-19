"""Command line interface: ``python -m explaincodebase <command> ...``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import load_env_file
from .llm import PROVIDER_CHOICES
from .methods import Settings
from .pipeline import ALL_METHODS, build_graph, input_packages, package_name, write_json

log = logging.getLogger("explaincodebase")


def _add_llm_options(parser: argparse.ArgumentParser, flag: str, help_role: str) -> None:
    parser.add_argument(flag, choices=PROVIDER_CHOICES, default=None,
                        help=f"LLM provider used {help_role} (default: vllm, or $ECB_PROVIDER)")
    parser.add_argument("--model", help="model name (default: $VLLM_MODEL, else the model the vLLM server serves)")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (default: $VLLM_API_BASE or http://localhost:8000/v1)")
    parser.add_argument("--no-cache", action="store_true", help="do not reuse cached LLM answers")
    parser.add_argument("--workers", type=int, default=8, help="parallel LLM requests (default: 8)")


def _methods(value: str) -> list[str]:
    if value == "all":
        return list(ALL_METHODS)
    chosen = [m.strip() for m in value.split(",") if m.strip()]
    unknown = sorted(set(chosen) - set(ALL_METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown method(s) {', '.join(unknown)}; choose from {', '.join(ALL_METHODS)}")
    return chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="explaincodebase", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output-dir", default="output", help="where results are written (default: output)")
    # The same options are accepted after the command name too.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--output-dir", default=argparse.SUPPRESS, help="where results are written (default: output)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("summarize", parents=[common], help="summarize every function/method of one or more codebases")
    p.add_argument("paths", nargs="*", help="codebase folders (default: every folder in input/)")
    _add_llm_options(p, "--provider", "for summarizing")
    p.add_argument("--methods", type=_methods, default=list(ALL_METHODS),
                   help=f"comma-separated subset of {','.join(ALL_METHODS)} or 'all' (default)")
    p.add_argument("--label", help="name for this run in output files (default: the model name)")
    p.add_argument("--languages", help="only these languages, comma-separated (see `languages`)")
    p.add_argument("--exclude", action="append", default=[], help="glob of files to skip (repeatable)")
    p.add_argument("--allow-imports", action="store_true",
                   help="docucp: import third-party Python packages to read their docs (runs their code)")
    p.add_argument("--max-code-chars", type=int, default=Settings.max_code_chars)
    p.add_argument("--baseline-max-chars", type=int, default=Settings.baseline_max_chars)
    p.add_argument("--skip-existing", action="store_true", help="skip methods whose output already exists")

    p = sub.add_parser("graph", parents=[common], help="only build and save the call graph")
    p.add_argument("paths", nargs="*")
    p.add_argument("--languages")
    p.add_argument("--exclude", action="append", default=[])

    p = sub.add_parser("evaluate", parents=[common], help="LLM-as-judge comparison of the generated summaries")
    p.add_argument("packages", nargs="*", help="package names or paths (default: all summarized packages)")
    _add_llm_options(p, "--judge", "as the judge")
    p.add_argument("--trials", type=int, default=10, help="judgements per unit (default: 10)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-modules", action="store_true", help="also judge top-level module code")
    p.add_argument("--source", help="codebase folder, if it moved since summarizing")

    p = sub.add_parser("mcq", parents=[common], help="answer mcqs/<package>.json using only each run's summaries")
    p.add_argument("packages", nargs="*")
    _add_llm_options(p, "--answerer", "to answer the questions")
    p.add_argument("--mcq-file", help="questions file (default: mcqs/<package>.json)")
    p.add_argument("--no-control", action="store_true", help="skip the no-documentation control")

    sub.add_parser("languages", parents=[common], help="list the languages this installation can analyse")
    return parser


def _languages(value: str | None) -> set[str] | None:
    return {v.strip() for v in value.split(",") if v.strip()} if value else None


def _paths(paths: list[str]) -> list[str]:
    if paths:
        missing = [p for p in paths if not os.path.isdir(p)]
        for path in missing:
            log.error("%s is not a directory; skipping it", path)
        return [p for p in paths if os.path.isdir(p)]
    found = input_packages()
    if not found:
        raise SystemExit("No codebase given and no folders in input/. Pass a path or copy a project into input/.")
    log.info("Processing every package in input/: %s", ", ".join(found))
    return found


def _summarized_packages(output_dir: str, requested: list[str]) -> list[str]:
    from .pipeline import load_runs

    if requested:
        return [package_name(p) for p in requested]
    packages = sorted({run["package"] for run in load_runs(output_dir)})
    if not packages:
        raise SystemExit(f"No summaries found in {output_dir}/summaries; run `summarize` first.")
    return packages


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not args.verbose:
        for noisy in ("httpx", "openai", "anthropic", "google_genai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "languages":
        from .languages import supported_languages

        for name, tier in supported_languages():
            print(f"{name:20} {tier}")
        return 0

    if args.command == "graph":
        failures = 0
        for path in _paths(args.paths):
            try:
                graph = build_graph(path, _languages(args.languages), args.exclude)
            except Exception as exc:
                log.error("%s: %s", path, exc, exc_info=args.verbose)
                failures += 1
                continue
            out = f"{args.output_dir}/graph/{package_name(path)}.json"
            data = graph.to_json()
            write_json(out, data)
            print(f"{out}: {json.dumps(data['stats'])}")
        return 1 if failures else 0

    if args.command == "summarize":
        from .pipeline import summarize_package

        failures = 0
        for path in _paths(args.paths):
            settings = Settings(workers=args.workers, allow_imports=args.allow_imports,
                                max_code_chars=args.max_code_chars,
                                baseline_max_chars=args.baseline_max_chars)
            try:
                written = summarize_package(
                    path, args.provider, args.methods, output_dir=args.output_dir, model=args.model,
                    base_url=args.base_url, label=args.label, settings=settings,
                    cache=not args.no_cache, languages=_languages(args.languages),
                    exclude=args.exclude, skip_existing=args.skip_existing)
            except Exception as exc:
                log.error("%s: %s", path, exc, exc_info=args.verbose)
                failures += 1
                continue
            for method, out in written.items():
                print(f"{method:9} {out}")
        return 1 if failures else 0

    if args.command == "evaluate":
        from .evaluate import evaluate_package

        failures = 0
        for package in _summarized_packages(args.output_dir, args.packages):
            try:
                result = evaluate_package(
                    package, output_dir=args.output_dir, judge=args.judge, model=args.model,
                    base_url=args.base_url, trials=args.trials, seed=args.seed, workers=args.workers,
                    include_modules=args.include_modules, source_path=args.source,
                    cache=not args.no_cache)
            except Exception as exc:
                log.error("%s: %s", package, exc, exc_info=args.verbose)
                failures += 1
                continue
            for warning in result["warnings"]:
                log.warning(warning)
            print(f"\n{package}: {result['units_evaluated']} units judged by {result['judge']['model']} -> {result['file']}")
            for key, row in result["by_run"].items():
                print(f"  {key:40} wins {row['wins']:5}  win rate {row['win_rate']:.1%}")
        return 1 if failures else 0

    if args.command == "mcq":
        from .mcq import run_mcq

        failures = 0
        for package in _summarized_packages(args.output_dir, args.packages):
            try:
                result = run_mcq(
                    package, mcq_file=args.mcq_file, output_dir=args.output_dir, answerer=args.answerer,
                    model=args.model, base_url=args.base_url, workers=args.workers,
                    control=not args.no_control, cache=not args.no_cache)
            except Exception as exc:
                log.error("%s: %s", package, exc, exc_info=args.verbose)
                failures += 1
                continue
            print(f"\n{package}: answered by {result['answerer']['model']} -> {result['file']}")
            for key, row in result["scores"].items():
                print(f"  {key:40} {row['correct']}/{row['total']}  ({row['accuracy']:.0%})")
        return 1 if failures else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
