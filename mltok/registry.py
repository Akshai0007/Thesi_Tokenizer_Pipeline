"""Plugin registry - the only thing the engine knows about plugins.

A plugin is ONE Python file that defines a class and registers it:

    from mltok.registry import register
    from mltok.contracts import Normalizer

    @register("normalizer", "my_norm")
    class MyNorm(Normalizer): ...

Discovery: every ``*.py`` under ``mltok/plugins/**`` and under each directory in
``plugins_dir`` (config) / ``$MLTOK_PLUGINS`` (os.pathsep separated) is imported.
Directories are loaded in that order and a later registration with the same
(kind, name) REPLACES the earlier one - so putting a file with the same name in
your own plugin directory replaces a built-in without touching the repo.

Components are referenced from the config as ``"name"`` or ``{type: name, **params}``.
Fingerprints (source-file hash + params) key every cache, so replacing a plugin file
automatically invalidates exactly the results that depended on it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

KINDS = (
    "source",  # where documents come from
    "filter",  # which documents are kept
    "normalizer",  # text normalization
    "pretokenizer",  # pre-tokenization
    "algorithm",  # tokenizer training algorithm
    "metric",  # intrinsic evaluation metrics
    "proxy_model",  # architecture of the proxy language model
    "selector",  # eligibility + ranking rules
    "store",  # where checkpoints are pushed to / restored from
)

_REG: dict[str, dict[str, type]] = {k: {} for k in KINDS}
_ERRORS: list[tuple[str, str]] = []
_OVERRIDES: list[str] = []
_LOADED: set[str] = set()
BUILTIN_DIR = Path(__file__).parent / "plugins"


def register(kind: str, name: str):
    if kind not in KINDS:
        raise ValueError(f"unknown plugin kind {kind!r}; choose from {KINDS}")

    def deco(cls):
        if name in _REG[kind] and _REG[kind][name] is not cls:
            old = getattr(_REG[kind][name], "_plugin_file", "?")
            _OVERRIDES.append(f"{kind}/{name}: {old} -> {inspect.getsourcefile(cls)}")
        cls._plugin_kind = kind
        cls._plugin_name = name
        cls._plugin_file = inspect.getsourcefile(cls)
        _REG[kind][name] = cls
        return cls

    return deco


def _import_file(path: Path) -> None:
    mod_name = "mltok_plugin_" + hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    if mod_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 - a broken plugin must not take the engine down
        sys.modules.pop(mod_name, None)
        _ERRORS.append((str(path), traceback.format_exc(limit=3)))


def _scan(root: Path) -> None:
    for p in sorted(root.rglob("*.py")):
        if p.name.startswith("_") or "__pycache__" in p.parts:
            continue
        _import_file(p)


def load_plugins(extra_dirs=()) -> None:
    """Idempotent. Built-ins first, then extra dirs in the given order (later wins)."""
    dirs = [BUILTIN_DIR] + [Path(d) for d in extra_dirs if d]
    env = os.environ.get("MLTOK_PLUGINS", "")
    dirs += [Path(d) for d in env.split(os.pathsep) if d]
    for d in dirs:
        key = str(d.resolve())
        if key in _LOADED or not d.exists():
            continue
        _LOADED.add(key)
        _scan(d)


def available(kind: str) -> list[str]:
    return sorted(_REG[kind])


def plugin_class(kind: str, name: str) -> type:
    try:
        return _REG[kind][name]
    except KeyError:
        raise KeyError(
            f"no {kind} plugin named {name!r}. Available: {available(kind)}. "
            f"Load errors: {[p for p, _ in _ERRORS] or 'none'}"
        ) from None


def spec_of(x) -> dict:
    """'ml' -> {'type': 'ml'};  {'type': 'ml', 'zw': 'drop'} -> same (copied)."""
    if isinstance(x, str):
        return {"type": x}
    if isinstance(x, dict) and "type" in x:
        return dict(x)
    raise ValueError(f"bad component spec {x!r}: use a name or a mapping with a 'type' key")


def make(kind: str, spec):
    s = spec_of(spec)
    name = s.pop("type")
    cls = plugin_class(kind, name)
    inst = cls(**s)
    inst.name, inst.params = name, s
    return inst


def fingerprint(kind: str, spec) -> str:
    """Hash of the plugin's source file + its parameters. Changes when either changes."""
    s = spec_of(spec)
    cls = plugin_class(kind, s["type"])
    src = Path(cls._plugin_file).read_bytes() if cls._plugin_file else b""
    blob = src + json.dumps(s, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def slug(spec) -> str:
    """Readable, filesystem-safe label: 'ml+zw-drop'."""
    s = spec_of(spec)
    extra = "".join(f"+{k}-{v}" for k, v in sorted(s.items()) if k != "type")
    return (s["type"] + extra).replace("/", "_").replace(" ", "")


def describe() -> list[dict]:
    rows = []
    for k in KINDS:
        for n, cls in sorted(_REG[k].items()):
            mod = sys.modules.get(cls.__module__)
            raw = cls.__dict__.get("__doc__") or (mod.__doc__ if mod else "") or ""
            doc = inspect.cleandoc(raw).splitlines()
            rows.append(
                {"kind": k, "name": n, "file": cls._plugin_file, "doc": doc[0] if doc else ""}
            )
    return rows


def load_errors() -> list[tuple[str, str]]:
    return list(_ERRORS)


def overrides() -> list[str]:
    return list(_OVERRIDES)


def keyhash(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"\x00")
    return h.hexdigest()[:12]
