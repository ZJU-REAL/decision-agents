"""Explicit model imports shared by applications and the training process.

To add a backend, implement its class in models/, register it at module scope,
and add its module path to MODEL_MODULES below. No training code needs changing.
Imports are lazy and idempotent; no model instances or connections are created.
"""
from importlib import import_module

from decision_model.registry import build_model


DEFAULT_MODEL = "fake_decisions_api"


MODEL_MODULES = (
    "decision_model.models.fake",
    "decision_model.models.decisions_api",
    "decision_model.models.prompted",
    "decision_model.models.constrained",
)


def load_models():
    """Execute each listed module's registration in the calling process."""
    for module in MODEL_MODULES:
        import_module(module)


def get_model(name=DEFAULT_MODEL, **options):
    """Load registered implementations, then construct the selected backend."""
    load_models()
    return build_model(name, **options)
