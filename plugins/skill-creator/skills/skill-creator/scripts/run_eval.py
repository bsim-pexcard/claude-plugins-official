#!/usr/bin/env python3
"""Run trigger evaluation for a skill description.

Tests whether a skill's description causes Claude to trigger (read the skill)
for a set of queries. Outputs results as JSON.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.utils import parse_skill_md


def find_project_root() -> Path:
    """Find the project root by walking up from cwd looking for .claude/.

    Mimics how Claude Code discovers its project root, so the command file
    we create ends up where claude -p will look for it.
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".claude").is_dir():
            return parent
    return current


def run_single_query(
    query: str,
    skill_name: str,
    skill_description: str,
    timeout: int,
    project_root: str,
    model: str | None = None,
) -> bool:
    """Run a single query and return whether the skill was triggered.

    Creates a real ephemeral SKILL (.claude/skills/<clean_name>/SKILL.md) with
    the candidate description so the model can actually invoke it via the Skill
    tool (commands in .claude/commands are user-slash-invoked and don't reliably
    surface to the model as invokable skills). Then runs `claude -p` with the raw
    query and watches the stream-json for a Skill/Read tool-use that references
    this skill's unique name. Detection spans the whole agentic turn — the model
    frequently writes a text preamble and invokes the skill in a LATER message —
    so only the terminal `result` event is treated as "did not trigger."
    """
    unique_id = uuid.uuid4().hex[:8]
    clean_name = f"{skill_name}-skill-{unique_id}"
    # Concurrency-safe trigger token. With num_workers>1, several ephemeral
    # skills with this SAME description coexist in the skills dir, all sharing
    # the "<skill_name>-skill-" prefix. The model may invoke ANY of them, so
    # detecting "the description triggered" means matching the shared prefix,
    # not this run's unique id (which would miscount ~(N-1)/N of real triggers).
    # Safe across candidates because run_loop evaluates descriptions serially.
    trigger_token = f"{skill_name}-skill-"
    project_skills_dir = Path(project_root) / ".claude" / "skills" / clean_name
    skill_file = project_skills_dir / "SKILL.md"

    try:
        project_skills_dir.mkdir(parents=True, exist_ok=True)
        # Use YAML block scalar to avoid breaking on quotes in description.
        indented_desc = "\n  ".join(skill_description.split("\n"))
        skill_content = (
            f"---\n"
            f"name: {clean_name}\n"
            f"description: |\n"
            f"  {indented_desc}\n"
            f"---\n\n"
            f"# {skill_name}\n\n"
            f"This skill handles: {skill_description}\n"
        )
        skill_file.write_text(skill_content, encoding="utf-8")

        cmd = [
            "claude",
            "-p", query,
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]
        if model:
            cmd.extend(["--model", model])

        # Remove CLAUDECODE env var to allow nesting claude -p inside a
        # Claude Code session. The guard is for interactive terminal conflicts;
        # programmatic subprocess usage is safe.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=project_root,
            env=env,
        )

        triggered = False
        start_time = time.time()
        # Track state for stream event detection
        pending_tool_name = None
        accumulated_json = ""
        recon_tools = ("Read", "Grep", "Glob", "LS")  # read-only; safe to let run

        # Read the subprocess pipe on a background thread and hand lines to the
        # main loop via a queue. select.select() is socket-only on Windows (it
        # raises WinError 10038 on a pipe handle), so we can't poll the pipe with
        # a timeout there. The queue lets us honor the overall timeout and
        # early-exit on trigger detection on every platform.
        import threading
        import queue as _queue
        _dbg = os.environ.get("SKILL_EVAL_DEBUG")  # optional path: dump raw stream-json for diagnosis
        line_q: "_queue.Queue" = _queue.Queue()

        def _reader():
            try:
                for raw in process.stdout:
                    s = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
                    if _dbg:
                        try:
                            with open(_dbg, "a", encoding="utf-8") as _fh:
                                _fh.write(s)
                        except Exception:
                            pass
                    line_q.put(s)
            finally:
                line_q.put(None)  # sentinel: stream closed

        threading.Thread(target=_reader, daemon=True).start()

        try:
            while time.time() - start_time < timeout:
                try:
                    line = line_q.get(timeout=1.0)
                except _queue.Empty:
                    continue
                if line is None:  # stream closed
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Detection spans the WHOLE agentic turn, but resolves as soon as
                # the model commits to a path — both to be fast and, importantly,
                # to stop BEFORE a sibling skill or a mutating tool actually runs.
                # These queries hit live ADO/repos, so a non-trigger case must not
                # be allowed to execute side effects (filing a bug, creating cases).
                #
                #   Skill(<clean_name>)          -> triggered            (True)
                #   Skill(<other>)               -> chose a sibling skill (False)
                #   Read/Grep/Glob/LS  (recon)   -> harmless; keep scanning
                #   any other tool (Bash/Edit/…) -> committed elsewhere   (False)
                #   terminal `result`            -> answered in text only (False)
                #
                # We deliberately do NOT treat a per-message boundary (message_stop)
                # as the end: the model routinely writes a text preamble and invokes
                # the skill in a LATER message.
                etype = event.get("type")
                if etype == "stream_event":
                    se = event.get("event", {})
                    se_type = se.get("type", "")

                    if se_type == "content_block_start":
                        cb = se.get("content_block", {})
                        if cb.get("type") == "tool_use":
                            name = cb.get("name", "")
                            accumulated_json = ""
                            if name in ("Skill", "Read"):
                                pending_tool_name = name      # accumulate input, match clean_name
                            elif name in recon_tools:
                                pending_tool_name = None       # read-only recon: ignore, keep scanning
                            else:
                                return False                   # Bash/Edit/Write/... -> not our skill

                    elif se_type == "content_block_delta" and pending_tool_name in ("Skill", "Read"):
                        delta = se.get("delta", {})
                        if delta.get("type") == "input_json_delta":
                            accumulated_json += delta.get("partial_json", "")
                            if trigger_token in accumulated_json:
                                return True

                    elif se_type == "content_block_stop":
                        # A Skill block that finished without matching trigger_token
                        # is a sibling skill -> not triggered. A Read of some other
                        # file is just recon -> keep scanning.
                        if pending_tool_name == "Skill":
                            return False
                        pending_tool_name = None
                        accumulated_json = ""

                # Fallback: full assistant message (fires once per message). Mirror
                # the same logic so a context-gathering Read isn't read as a miss.
                elif etype == "assistant":
                    message = event.get("message", {})
                    for content_item in message.get("content", []):
                        if content_item.get("type") != "tool_use":
                            continue
                        name = content_item.get("name", "")
                        inp = content_item.get("input", {})
                        if name == "Skill":
                            blob = str(inp.get("skill", "")) + " " + str(inp.get("command", ""))
                            return trigger_token in blob
                        if name == "Read":
                            if trigger_token in str(inp.get("file_path", "")):
                                return True
                            continue
                        if name in recon_tools:
                            continue
                        return False

                # Terminal: the agentic turn finished without invoking the skill.
                elif etype == "result":
                    return triggered
        finally:
            # Clean up process on any exit path (return, exception, timeout)
            if process.poll() is None:
                process.kill()
                process.wait()

        return triggered
    finally:
        try:
            if skill_file.exists():
                skill_file.unlink()
            if project_skills_dir.exists():
                project_skills_dir.rmdir()
        except OSError:
            pass


def run_eval(
    eval_set: list[dict],
    skill_name: str,
    description: str,
    num_workers: int,
    timeout: int,
    project_root: Path,
    runs_per_query: int = 1,
    trigger_threshold: float = 0.5,
    model: str | None = None,
) -> dict:
    """Run the full eval set and return results."""
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_info = {}
        for item in eval_set:
            for run_idx in range(runs_per_query):
                future = executor.submit(
                    run_single_query,
                    item["query"],
                    skill_name,
                    description,
                    timeout,
                    str(project_root),
                    model,
                )
                future_to_info[future] = (item, run_idx)

        query_triggers: dict[str, list[bool]] = {}
        query_items: dict[str, dict] = {}
        for future in as_completed(future_to_info):
            item, _ = future_to_info[future]
            query = item["query"]
            query_items[query] = item
            if query not in query_triggers:
                query_triggers[query] = []
            try:
                query_triggers[query].append(future.result())
            except Exception as e:
                print(f"Warning: query failed: {e}", file=sys.stderr)
                query_triggers[query].append(False)

    for query, triggers in query_triggers.items():
        item = query_items[query]
        trigger_rate = sum(triggers) / len(triggers)
        should_trigger = item["should_trigger"]
        if should_trigger:
            did_pass = trigger_rate >= trigger_threshold
        else:
            did_pass = trigger_rate < trigger_threshold
        results.append({
            "query": query,
            "should_trigger": should_trigger,
            "trigger_rate": trigger_rate,
            "triggers": sum(triggers),
            "runs": len(triggers),
            "pass": did_pass,
        })

    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    return {
        "skill_name": skill_name,
        "description": description,
        "results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run trigger evaluation for a skill description")
    parser.add_argument("--eval-set", required=True, help="Path to eval set JSON file")
    parser.add_argument("--skill-path", required=True, help="Path to skill directory")
    parser.add_argument("--description", default=None, help="Override description to test")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per query in seconds")
    parser.add_argument("--runs-per-query", type=int, default=3, help="Number of runs per query")
    parser.add_argument("--trigger-threshold", type=float, default=0.5, help="Trigger rate threshold")
    parser.add_argument("--model", default=None, help="Model to use for claude -p (default: user's configured model)")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text())
    skill_path = Path(args.skill_path)

    if not (skill_path / "SKILL.md").exists():
        print(f"Error: No SKILL.md found at {skill_path}", file=sys.stderr)
        sys.exit(1)

    name, original_description, content = parse_skill_md(skill_path)
    description = args.description or original_description
    project_root = find_project_root()

    if args.verbose:
        print(f"Evaluating: {description}", file=sys.stderr)

    output = run_eval(
        eval_set=eval_set,
        skill_name=name,
        description=description,
        num_workers=args.num_workers,
        timeout=args.timeout,
        project_root=project_root,
        runs_per_query=args.runs_per_query,
        trigger_threshold=args.trigger_threshold,
        model=args.model,
    )

    if args.verbose:
        summary = output["summary"]
        print(f"Results: {summary['passed']}/{summary['total']} passed", file=sys.stderr)
        for r in output["results"]:
            status = "PASS" if r["pass"] else "FAIL"
            rate_str = f"{r['triggers']}/{r['runs']}"
            print(f"  [{status}] rate={rate_str} expected={r['should_trigger']}: {r['query'][:70]}", file=sys.stderr)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
