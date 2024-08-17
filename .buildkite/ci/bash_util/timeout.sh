#!/usr/bin/env bash

run_with_timeout() {
  # Helper bash function which runs a command with a timeout.
  # It will run the process in the foreground (so signals sent to the test
  # runner are forwarded to the pytest process), and it will print when the
  # command times out or succeeds.
  #
  # It also unsets and resets errexit.

  # Capture the current state of 'errexit' (set -e)
  local errexit_status=$(shopt -o errexit | awk '{print $2}')

  # Disable 'exit on error' temporarily for the scope of this function
  set +e

  local duration=$1
  shift
  timeout --foreground "$duration" "$@"
  local status=$?

  # Restore the original 'errexit' state
  if [[ $errexit_status == "on" ]]; then
    set -e
  else
    set +e
  fi

  # Check for timeout
  if [ $status -eq 124 ]; then
    echo "Command timed out: $*"
    exit 124
  elif [ $status -ne 0 ]; then
    echo "Command failed with status $status: $*"
    exit $status
  else
    echo "Command completed with exitcode 0: $*"
  fi
}
