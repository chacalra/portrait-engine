"""Print the JSON contracts a tool-calling agent would receive.

This is the entire interface the future agent needs from this repository. The contracts are
generated from each tool's signature and docstring, so they cannot drift from the code.

    python examples/03_tool_contracts.py
"""

from __future__ import annotations

import json

from portrait_engine import tool_schemas

if __name__ == "__main__":
    print(json.dumps(tool_schemas(), indent=2))
