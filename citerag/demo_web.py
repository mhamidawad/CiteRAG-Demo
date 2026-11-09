"""Public demo entrypoint for the Northflank Sandbox deployment.

The production entrypoint (``python -m citerag.web``) refuses a non-loopback
listener unless a bearer token is configured, and the dashboard cannot send a
bearer token from a browser. This entrypoint deliberately serves the same
read-only application without that guard so a public demo can be opened.

It is limited to public, non-sensitive project documentation and must not be
used with private data, provider credentials, or production traffic.
"""
import os

import uvicorn

from citerag.semantic import model
from citerag.web import app


def validate_demo_configuration() -> None:
    """Fail closed unless the operator explicitly opted into the public demo."""
    if os.getenv("CITERAG_DEMO_PUBLIC") != "1":
        raise SystemExit("CITERAG_DEMO_PUBLIC=1 is required for the demo entrypoint")
    if os.getenv("CITERAG_PRODUCTION") == "1" or os.getenv("CITERAG_REQUIRE_AUTH") == "1":
        raise SystemExit("The demo entrypoint cannot run in production/auth mode")
    if os.getenv("CITERAG_API_TOKEN"):
        raise SystemExit("The demo entrypoint does not accept a bearer token")


def main() -> None:
    validate_demo_configuration()
    # Load the pre-baked model before accepting traffic so a broken cache fails
    # the rollout instead of the first user question.
    model()
    uvicorn.run(
        app,
        host=os.getenv("CITERAG_WEB_HOST", "0.0.0.0"),
        port=int(os.getenv("CITERAG_WEB_PORT", "18762")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
