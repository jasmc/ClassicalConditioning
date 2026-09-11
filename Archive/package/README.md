# Archived Package Legacy Execution

This directory preserves the retired package legacy implementation and its
tests as read-only source history. Its `src/classical_conditioning` layout
matches the former installable package paths to make review and Git history
straightforward.

It is deliberately outside the active `src/` tree, is not installed by the
package build, and is not collected by the normal test suite. Do not add
imports from active package code to this archive or treat it as a supported
workflow. The supported package route is the candidate pipeline.
