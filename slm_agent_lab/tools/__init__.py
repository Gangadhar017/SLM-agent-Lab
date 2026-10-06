"""Four deterministic tools. Determinism matters: every task has an exact ground truth, so scoring is objective."""

from .registry import TOOL_NAMES, TOOL_SCHEMAS, ToolError, ToolResult, execute_tool  # noqa: F401
