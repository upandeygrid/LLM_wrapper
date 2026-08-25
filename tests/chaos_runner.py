"""
LLM Shield — Chaos Runner CLI.

Fires N requests against the Shield library with ChaosProvider active,
then verifies that all production invariants hold even under fault injection.

Usage:
    python -m tests.chaos_runner                   # 20 requests, default rates
    python -m tests.chaos_runner --requests 100    # 100 requests
    python -m tests.chaos_runner --seed 42         # reproducible run
    python -m tests.chaos_runner --fault-rate 0.5  # 50% of calls get a fault

    OR via Makefile:
    make chaos
    make chaos ARGS="--requests 100 --seed 42"

This script NEVER runs automatically. It is a deliberate opt-in test tool.
It is NOT imported by any production module.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Invariant definitions — what MUST always be true
# ---------------------------------------------------------------------------

INVARIANTS = [
    "Every execution terminates in SUCCEEDED or FAILED (never hangs)",
    "execution_trace is always populated",
    "FAILED responses always carry an escalation packet",
    "SUCCEEDED responses always have a non-None result",
    "states_visited is never empty",
    "total_llm_calls >= 1 for every request",
    "No unhandled exceptions escape the Shield engine",
]


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class ChaosRunResult:
    total: int
    succeeded: int = 0
    escalated: int = 0          # FAILED with proper escalation packet
    invariant_violations: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Invariant checker
# ---------------------------------------------------------------------------

def check_invariants(response: object, request_index: int) -> list[str]:
    """
    Verify all production invariants on a ShieldResponse.
    Returns a list of violation messages (empty = all passed).
    """
    violations = []
    prefix = f"Request #{request_index}"

    # Invariant 1: terminal status
    if response.status not in ("SUCCEEDED", "FAILED"):
        violations.append(f"{prefix}: unexpected status '{response.status}'")

    # Invariant 2: trace always present
    if response.execution_trace is None:
        violations.append(f"{prefix}: execution_trace is None")
        return violations  # can't check further without trace

    trace = response.execution_trace

    # Invariant 3: FAILED must have escalation packet
    if response.status == "FAILED" and response.escalation is None:
        violations.append(f"{prefix}: FAILED response missing escalation packet")

    # Invariant 4: SUCCEEDED must have result
    if response.status == "SUCCEEDED" and response.result is None:
        violations.append(f"{prefix}: SUCCEEDED response has None result")

    # Invariant 5: states_visited never empty
    if not trace.states_visited:
        violations.append(f"{prefix}: states_visited is empty")

    # Invariant 6: at least one LLM call recorded
    if trace.total_llm_calls < 1:
        violations.append(f"{prefix}: total_llm_calls < 1")

    return violations


# ---------------------------------------------------------------------------
# Test scenarios — varied prompts and schemas for realistic coverage
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "prompt": "Generate a JSON user profile with name and age.",
        "schema": {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
        },
    },
    {
        "prompt": "Generate a JSON product with name and price.",
        "schema": {
            "type": "object",
            "required": ["name", "price"],
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number", "minimum": 0},
            },
        },
    },
    {
        "prompt": "Generate a JSON movie entry with title and year.",
        "schema": {
            "type": "object",
            "required": ["title", "year"],
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer"},
            },
        },
    },
    {
        "prompt": "Generate a JSON status report with status and code.",
        "schema": {
            "type": "object",
            "required": ["status", "code"],
            "properties": {
                "status": {"type": "string"},
                "code": {"type": "integer"},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Core chaos execution loop
# ---------------------------------------------------------------------------

async def run_chaos(args: argparse.Namespace) -> ChaosRunResult:
    # --- Imports are local so this module never accidentally loads in production ---
    from llm_shield import Shield, ShieldConfig, ShieldRequest
    from llm_shield.chaos import ChaosConfig, ChaosProvider
    from llm_shield.providers import LiteLLMProvider

    # Split the configured fault_rate evenly across fault types
    per_fault = args.fault_rate / 6.0
    chaos_config = ChaosConfig(
        timeout_rate=per_fault,
        server_error_rate=per_fault,
        rate_limit_rate=per_fault,
        malformed_json_rate=per_fault,
        empty_response_rate=per_fault,
        truncated_json_rate=per_fault,
        seed=args.seed,
    )

    # ── Opt-in gate: ChaosProvider is constructed HERE, explicitly ──
    chaos_provider = ChaosProvider(
        base=LiteLLMProvider(),
        config=chaos_config,
    )

    shield = Shield(
        config=ShieldConfig(
            max_retries=args.max_retries,
            max_repairs=args.max_repairs,
            timeout_seconds=args.timeout,
        ),
        provider=chaos_provider,
    )

    results = ChaosRunResult(total=args.requests)
    start = time.perf_counter()

    import random
    rng = random.Random(args.seed)

    for i in range(args.requests):
        scenario = rng.choice(SCENARIOS)
        idx = i + 1

        if args.verbose:
            print(f"  [{idx}/{args.requests}] ", end="", flush=True)

        try:
            response = await shield.execute(ShieldRequest(
                prompt=scenario["prompt"],
                response_schema=scenario["schema"],
            ))

            violations = check_invariants(response, idx)

            if violations:
                results.invariant_violations += len(violations)
                results.errors.extend(violations)
                if args.verbose:
                    print(f"INVARIANT VIOLATION ✗")
            elif response.status == "SUCCEEDED":
                results.succeeded += 1
                if args.verbose:
                    print(f"SUCCEEDED ✓")
            else:
                results.escalated += 1
                if args.verbose:
                    print(f"ESCALATED (controlled failure) ✓")

        except Exception as exc:
            # An unhandled exception escaping the engine is itself an invariant violation
            msg = f"Request #{idx}: UNHANDLED EXCEPTION — {type(exc).__name__}: {exc}"
            results.invariant_violations += 1
            results.errors.append(msg)
            if args.verbose:
                print(f"UNHANDLED EXCEPTION ✗")

    results.duration_s = time.perf_counter() - start
    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results: ChaosRunResult, fault_rate: float) -> None:
    width = 60
    print("\n" + "=" * width)
    print("  LLM SHIELD — CHAOS TEST REPORT")
    print("=" * width)
    print(f"  Total Requests    : {results.total}")
    print(f"  Configured Fault  : {fault_rate * 100:.0f}% of LLM calls get a fault")
    print(f"  ─────────────────────────────────────────")
    print(f"  ✓ SUCCEEDED       : {results.succeeded}")
    print(f"  ✓ ESCALATED (ok)  : {results.escalated}  ← controlled failures")
    print(f"  ✗ INVARIANT FAILS : {results.invariant_violations}  ← MUST BE 0")
    print(f"  Duration          : {results.duration_s:.2f}s")
    print("-" * width)

    if results.errors:
        print("\n  INVARIANT VIOLATIONS DETAIL:")
        for err in results.errors:
            print(f"  ✗ {err}")
        print()

    if results.invariant_violations == 0:
        print(
            f"  ✅  ALL INVARIANTS PASSED\n"
            f"      {results.total} requests under {fault_rate*100:.0f}% fault injection.\n"
            f"      llm_shield is resilient."
        )
    else:
        print(
            f"  ❌  {results.invariant_violations} INVARIANT(S) VIOLATED\n"
            f"      Review violations above — the engine has a bug."
        )

    print("=" * width)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tests.chaos_runner",
        description=(
            "LLM Shield Chaos Runner — deliberate fault injection testing.\n"
            "Verifies that llm_shield always terminates cleanly, never crashes,\n"
            "and never silently drops errors, even under heavy fault injection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--requests", type=int, default=20,
        help="Number of requests to fire (default: 20)",
    )
    parser.add_argument(
        "--fault-rate", type=float, default=0.4,
        help=(
            "Fraction of LLM calls that receive a deliberate fault "
            "(0.0–1.0, default: 0.4). Split evenly across 6 fault types."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible fault sequences (default: random)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Shield max retries per model (default: 2)",
    )
    parser.add_argument(
        "--max-repairs", type=int, default=1,
        help="Shield max repair attempts (default: 1)",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help=(
            "Per-call timeout in seconds (default: 5.0). "
            "Keep low during chaos runs so timeout faults resolve quickly."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-request status",
    )

    args = parser.parse_args()

    if not 0.0 <= args.fault_rate <= 1.0:
        parser.error("--fault-rate must be between 0.0 and 1.0")

    print("\n" + "=" * 60)
    print("  LLM SHIELD — CHAOS TESTING MODE")
    print("  ⚠️  Fault injection ACTIVE — not for production use")
    print("=" * 60)
    print(f"  Requests    : {args.requests}")
    print(f"  Fault Rate  : {args.fault_rate * 100:.0f}% of calls get a fault")
    print(f"  Seed        : {args.seed or 'random'}")
    print(f"  Max Retries : {args.max_retries}")
    print(f"  Max Repairs : {args.max_repairs}")
    print(f"  Timeout     : {args.timeout}s per call")
    print("-" * 60)

    print("\n  Invariants being verified:")
    for inv in INVARIANTS:
        print(f"    • {inv}")

    print("\n  Running...\n")

    results = asyncio.run(run_chaos(args))
    print_report(results, args.fault_rate)

    sys.exit(1 if results.invariant_violations > 0 else 0)


if __name__ == "__main__":
    main()
