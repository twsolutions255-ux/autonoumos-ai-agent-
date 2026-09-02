"""Tool modules.

Importing a module is what registers its tools — the decorators run at import
time. The brain loop imports the original six by name; these two are imported
here instead, so that anything which touches `atlas.tools` at all (the loop,
the console, the tests) gets the complete surface rather than a subset that
depends on which module happened to be named at the call site.
"""
from . import builder, outreach, pipeline  # noqa: F401
