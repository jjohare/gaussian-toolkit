# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""CLI entry point: python -m pipeline.cli video2scene <video> <output> [options]"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.orchestrator import PipelineOrchestrator, PipelineState


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Gaussian Toolkit video-to-scene agentic pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- video2scene ---
    v2s = sub.add_parser("video2scene", help="Run the full video-to-scene pipeline")
    v2s.add_argument("video", help="Path to input video file")
    v2s.add_argument("output", help="Output directory")
    v2s.add_argument("--config", "-c", help="Path to pipeline config JSON")
    v2s.add_argument("--fps", type=float, help="Frame extraction FPS")
    v2s.add_argument("--max-iter", type=int, help="Max training iterations")
    v2s.add_argument("--strategy", choices=["mcmc", "default"], help="Training strategy")
    v2s.add_argument("--endpoint", help="MCP endpoint URL")
    v2s.add_argument("--objects", nargs="*", help="Object descriptions for decomposition")
    v2s.add_argument("--target-psnr", type=float, help="Target PSNR threshold")
    v2s.add_argument("--max-retries", type=int, help="Max retries per stage")
    v2s.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    v2s.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    # --- config ---
    cfg_cmd = sub.add_parser("config", help="Manage pipeline configuration")
    cfg_sub = cfg_cmd.add_subparsers(dest="config_action", required=True)

    cfg_show = cfg_sub.add_parser("show", help="Show default config")
    cfg_show.add_argument("--output", "-o", help="Write to file instead of stdout")

    cfg_validate = cfg_sub.add_parser("validate", help="Validate a config file")
    cfg_validate.add_argument("file", help="Config file to validate")

    # --- status ---
    st = sub.add_parser("status", help="Show pipeline status from a status file")
    st.add_argument("status_file", help="Path to pipeline_status.json")

    return parser


def _apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> None:
    if args.fps is not None:
        config.ingest.fps = args.fps
    if args.max_iter is not None:
        config.training.max_iterations = args.max_iter
    if args.strategy is not None:
        config.training.strategy = args.strategy
    if args.endpoint is not None:
        config.mcp_endpoint = args.endpoint
    if args.objects is not None:
        config.decompose.descriptions = args.objects
    if args.target_psnr is not None:
        config.training.target_psnr = args.target_psnr
        config.quality.gate1_min_psnr = args.target_psnr * 0.8
    if args.max_retries is not None:
        config.retry.max_retries = args.max_retries


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _progress_display(status_file: Path) -> None:
    """Print a simple progress bar from the status file."""
    try:
        data = json.loads(status_file.read_text(encoding="utf-8"))
        progress = data.get("progress", 0.0)
        state = data.get("state", "UNKNOWN")
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stderr.write(f"\r[{bar}] {progress:.0%} {state}")
        sys.stderr.flush()
    except (OSError, json.JSONDecodeError):
        pass


def cmd_video2scene(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.quiet)

    if args.config:
        config = PipelineConfig.load(args.config)
    else:
        config = PipelineConfig()

    _apply_overrides(config, args)

    errors = config.validate()
    if errors:
        for err in errors:
            print(f"Config error: {err}", file=sys.stderr)
        return 1

    video = Path(args.video)
    if not video.exists():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1

    output = Path(args.output)
    orchestrator = PipelineOrchestrator(
        video_path=str(video),
        output_dir=str(output),
        config=config,
    )

    print(f"Pipeline: {video} -> {output}")
    print(f"Strategy: {config.training.strategy}, max_iter: {config.training.max_iterations}")

    start = time.monotonic()
    status = orchestrator.run()
    elapsed = time.monotonic() - start

    print()  # newline after progress bar
    if status.state == PipelineState.DONE.value:
        print(f"Pipeline completed in {elapsed:.1f}s")
        partial = orchestrator.get_partial_results()
        for name, path in partial["artifacts"].items():
            if path:
                print(f"  {name}: {path}")
        return 0
    else:
        print(f"Pipeline FAILED at state {status.state} after {elapsed:.1f}s")
        if status.error:
            print(f"  Error: {status.error}")
        partial = orchestrator.get_partial_results()
        if partial["artifacts"]:
            print("  Partial artifacts:")
            for name, path in partial["artifacts"].items():
                if path:
                    print(f"    {name}: {path}")
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    if args.config_action == "show":
        config = PipelineConfig()
        text = json.dumps(config.to_dict(), indent=2, default=str)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Config written to {args.output}")
        else:
            print(text)
        return 0

    elif args.config_action == "validate":
        try:
            config = PipelineConfig.load(args.file)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Error loading config: {exc}", file=sys.stderr)
            return 1

        errors = config.validate()
        if errors:
            for err in errors:
                print(f"  FAIL: {err}", file=sys.stderr)
            return 1
        print("Config is valid.")
        return 0

    return 1


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.status_file)
    if not path.exists():
        print(f"Status file not found: {path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "video2scene": cmd_video2scene,
        "config": cmd_config,
        "status": cmd_status,
    }
    handler = dispatch.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
