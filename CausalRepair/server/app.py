# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Causalrepair Environment.

This module creates an HTTP server that exposes the CausalrepairEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Adapter selection (set CR_ADAPTER before starting the server):
    "code"      — CodeRepairAdapter (3 functions, exec-based tests)  [DEFAULT]
    "hydraulic" — MockAdapter (valve -> pressure -> alarm)

Usage:
    # Code repair (default):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Hydraulic domain:
    $env:CR_ADAPTER="hydraulic"; uvicorn server.app:app --port 8000   # PowerShell
    CR_ADAPTER=hydraulic uvicorn server.app:app --port 8000           # bash/Linux

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e


import os

from models import CausalrepairAction, CausalrepairObservation
from server.CausalRepair_environment import CausalrepairEnvironment
from server.mock_adapter import MockAdapter
from server.code_repair_adapter import CodeRepairAdapter

# Registry: CR_ADAPTER value -> adapter class.
# To add a new domain, register it here in one line.
ADAPTERS = {
    "code":      CodeRepairAdapter,
    "hydraulic": MockAdapter,
}

_adapter_name = os.environ.get("CR_ADAPTER", "code").lower()
if _adapter_name not in ADAPTERS:
    raise ValueError(
        f"Unknown CR_ADAPTER={_adapter_name!r}. "
        f"Valid choices: {sorted(ADAPTERS)}"
    )
_adapter_cls = ADAPTERS[_adapter_name]
print(f"[CausalRepair] Using adapter: {_adapter_name} ({_adapter_cls.__name__})")
# Create a single persistent environment instance
persistent_env = CausalrepairEnvironment(adapter=_adapter_cls())

# Always return the same instance for every request/session
app = create_app(
    lambda: persistent_env,
    CausalrepairAction,
    CausalrepairObservation,
    env_name="CausalRepair",
    max_concurrent_envs=1,  # increase to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m CausalRepair.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn CausalRepair.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
