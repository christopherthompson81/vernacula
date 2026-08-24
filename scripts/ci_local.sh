#!/usr/bin/env bash
#
# Dev-side CI. Runs the whole test suite the way hosted CI cannot: on a machine
# that actually has the ONNX model packages and the test_audio/ tree.
#
# The point of this script is the skip check. Hosted CI runs the asset-gated
# tests too, but with no models present they report 23 skips and a green tick --
# a step that passes unconditionally is not coverage. Here the assets exist, so
# a skip means the resolver stopped finding them, and that is a failure.
#
# Usage:
#   scripts/ci_local.sh              # build, then run every test project
#   scripts/ci_local.sh --no-build   # reuse the existing bin/x64 artifacts
#
# Set VERNACULA_CI_ALLOW_SKIPS=1 to downgrade the skip check to a warning, for
# running this on a box that legitimately has no model assets.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUILD=1
[[ "${1:-}" == "--no-build" ]] && BUILD=0

# Tests run against the artifacts the solution build produced (bin/x64/...).
# Without -p:Platform=x64 dotnet test rebuilds into bin/ instead, so the thing
# tested is not the thing built.
TEST_ARGS=(--no-build -p:Platform=x64 -p:EP=Cpu)

# project<TAB>needs_assets
#
# Only mark a project asset-gated when *every* skip in it means "the resolver
# stopped finding the assets". Projects that also carry opt-in-heavy tests
# (guarded by their own env var rather than by asset presence) do not qualify:
# their skips are a deliberate default, not rot.
PROJECTS=(
  "tests/Vernacula.Tests	0"
  "tests/Chatterbox.Tests	0"
  "tests/AsrBackendCoverage	0"
  "tests/IndicConformerTest	1"
)

if (( BUILD )); then
  echo "==> Building Vernacula.slnx (EP=Cpu)"
  if ! dotnet build Vernacula.slnx -p:EP=Cpu; then
    echo "FAIL: solution build" >&2
    exit 1
  fi
fi

failures=()

for entry in "${PROJECTS[@]}"; do
  IFS=$'\t' read -r proj needs_assets <<<"$entry"
  echo
  echo "==> $proj"

  out=$(dotnet test "$proj" "${TEST_ARGS[@]}" 2>&1)
  status=$?
  echo "$out" | tail -3

  if (( status != 0 )); then
    echo "$out" >&2
    failures+=("$proj: test run failed")
    continue
  fi

  # `dotnet test --no-build` against a project that was never built prints
  # nothing and exits 0 -- a green run of zero tests. Nothing downstream can
  # tell that apart from success, so require a summary line before believing
  # any of this.
  if ! grep -qE '^(Passed!|Failed!|Skipped!)' <<<"$out"; then
    failures+=("$proj: no test summary -- nothing ran (missing build? wrong -p:Platform?)")
    continue
  fi

  # dotnet test exits 0 when every test skips, so the summary line is the only
  # place the distinction between "ran" and "did not run" survives.
  if (( needs_assets )) && ! grep -qE 'Skipped: +0,' <<<"$out"; then
    skipped=$(grep -oE 'Skipped: +[0-9]+' <<<"$out" | head -1)
    if [[ -n "${VERNACULA_CI_ALLOW_SKIPS:-}" ]]; then
      echo "WARN: $proj $skipped (assets missing; allowed by VERNACULA_CI_ALLOW_SKIPS)"
    else
      failures+=("$proj: $skipped -- asset-gated tests did not run on a box that has the assets")
    fi
  fi
done

echo
if (( ${#failures[@]} )); then
  echo "FAILED:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
fi
echo "All test projects passed."
