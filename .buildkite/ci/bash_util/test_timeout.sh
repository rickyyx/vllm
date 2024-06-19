#!/usr/bin/env bash

source .buildkite/ci/bash_util/timeout.sh

# Test 1: Command completes within the timeout
test_command_completes() {
  output=$(run_with_timeout 2 sleep 1)
  if [[ $? -eq 0 && "$output" =~ "Command completed with exitcode 0: sleep 1" ]]; then
    echo "Test 1 Passed: Command completed within the timeout."
  else
    echo "Test 1 Failed: Unexpected output or error status."
    exit 1
  fi
}

# Test 2: Command does not complete within timeout
test_command_timeout() {
  output=$(run_with_timeout 1 sleep 2 2>&1)
  if [[ $? -eq 124 && "$output" =~ "Command timed out: sleep 2" ]]; then
    echo "Test 2 Passed: Command timed out as expected."
  else
    echo "Test 2 Failed: Wrong exit status or message. Received: $output"
    exit 1
  fi
}

# Test 3: Command fails
test_command_fails() {
  output=$(run_with_timeout 2 false 2>&1)
  if [[ $? -ne 0 && $? -ne 124 && "$output" =~ "Command failed with status" ]]; then
    echo "Test 3 Passed: Command failed as expected. Message: $output"
  else
    echo "Test 3 Failed: Unexpected success or wrong failure message."
    exit 1
  fi
}

# Execute tests

echo "Running bash util tests"

set -ux

test_command_completes
test_command_timeout
test_command_fails

echo "All bash util tests passed."
