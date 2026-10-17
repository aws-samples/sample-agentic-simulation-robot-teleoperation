"""Let the agent package import on a machine without Strands (unit tests only exercise plain code).

Strands requires Python >= 3.10 and Bedrock access; neither is needed to test the protocol, the
geometry or the text the tools produce. When ``strands`` is importable the shim does nothing.
"""
import sys
import types

try:
    import strands  # noqa: F401
except ImportError:  # pragma: no cover - only on hosts without strands
    shim = types.ModuleType("strands")

    def tool(fn=None, **_kw):
        return fn if fn is not None else (lambda f: f)

    shim.tool = tool
    sys.modules["strands"] = shim
