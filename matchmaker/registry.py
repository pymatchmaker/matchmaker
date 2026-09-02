"""Declarative registry of score-following methods and feature processors.

The registry data lives in :data:`SPEC_FILE` (``matchmaker/methods.yaml``);
this module is the interpreter for it. Adding a method or a processor to
Matchmaker is therefore an edit to that YAML file, not to
:mod:`matchmaker.matchmaker` -- as long as every constructor argument can be
named with the vocabulary below.

Three small vocabularies connect the YAML to the code:

``PROVIDERS``
    ``{from: <name>}`` values. Each is a function of the ``Matchmaker``
    instance, registered with :func:`provider`.
``REFERENCE_BUILDERS``
    ``reference: <name>`` on a method: how the score-side features are built.
    Registered with :func:`reference_builder`.
``!obj pkg.module:Attribute``
    An imported Python object, for defaults that must stay tied to a constant
    or class defined in code.

Anything a method needs that none of these express belongs in a new provider
or reference builder here -- keep the YAML free of Python.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import yaml

__all__ = [
    "SPEC_FILE",
    "PROVIDERS",
    "REFERENCE_BUILDERS",
    "MethodSpec",
    "ProcessorSpec",
    "Registry",
    "REGISTRY",
    "load_object",
    "provider",
    "reference_builder",
]

SPEC_FILE = Path(__file__).parent / "methods.yaml"

#: Keys that mark a mapping in the YAML as an argument spec rather than a
#: literal value. A mapping using exactly one of them is resolved; any other
#: mapping is passed through as-is.
_ARG_SOURCES = ("from", "config", "value")


# ---------------------------------------------------------------------------
# Importing objects named in the spec
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def load_object(path: str) -> Any:
    """Import the object named by a ``"package.module:Attribute"`` path."""
    if ":" not in path:
        raise ValueError(
            f"Invalid object path '{path}'. Expected 'package.module:Attribute'."
        )
    module_name, _, attr = path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import '{module_name}' for '{path}': {e}") from e
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"'{module_name}' has no attribute '{attr}'.") from e


class _SpecLoader(yaml.SafeLoader):
    """SafeLoader plus the ``!obj`` tag, which imports the named object."""


_SpecLoader.add_constructor(
    "!obj", lambda loader, node: load_object(loader.construct_scalar(node))
)


# ---------------------------------------------------------------------------
# Providers: values read off a live Matchmaker instance
# ---------------------------------------------------------------------------
PROVIDERS: Dict[str, Callable[[Any], Any]] = {}


def provider(name: str) -> Callable:
    """Register a ``{from: <name>}`` value source."""

    def decorate(func):
        PROVIDERS[name] = func
        return func

    return decorate


provider("reference_features")(lambda mm: mm.reference_features)
provider("processor")(lambda mm: mm.processor)
provider("queue")(lambda mm: mm.stream.queue)
provider("score_part")(lambda mm: mm.score_part)
provider("score_positions")(lambda mm: mm.score_positions)
provider("note_array")(lambda mm: mm.score_part.note_array())
provider("note_array_with_grace")(
    lambda mm: mm.score_part.note_array(include_grace_notes=True)
)
provider("tempo")(lambda mm: mm.tempo)
provider("frame_rate")(lambda mm: mm.frame_rate)
provider("sample_rate")(lambda mm: mm.sample_rate)
provider("hop_length")(lambda mm: mm.hop_length)
provider("polling_period")(lambda mm: mm.polling_period)
provider("ref_frame_to_beat")(lambda mm: mm.ref_frame_to_beat())
provider("performance_file")(lambda mm: mm.performance_file)
provider("method")(lambda mm: mm.method)
provider("input_type")(lambda mm: mm.input_type)


@provider("audio_hop_seconds")
def _audio_hop_seconds(mm) -> float:
    """Duration of one audio hop, in seconds."""
    return mm.hop_length / mm.sample_rate


@provider("default_polling_period")
def _default_polling_period(mm) -> float:
    """The MidiStream default window, in seconds."""
    from matchmaker.io.midi import POLLING_PERIOD

    return POLLING_PERIOD


@provider("reference_sample_rate")
def _reference_sample_rate(mm) -> int:
    """Sample rate for score-side rendering (MIDI input has no stream rate)."""
    from matchmaker.features.audio import SAMPLE_RATE

    return mm.sample_rate if mm.input_type == "audio" else SAMPLE_RATE


def _onset_pianoroll(mm):
    """``(features, score_positions)`` of the score's onset pianoroll, memoised."""
    cached = getattr(mm, "_onset_pianoroll_cache", None)
    if cached is None:
        from matchmaker.features.midi import onset_pianoroll

        cached = onset_pianoroll(
            mm.reference_features,
            onset_key="onset_beat",
            piano_range=mm.config.get("piano_range", True),
        )
        mm._onset_pianoroll_cache = cached
    return cached


provider("onset_pianoroll_features")(lambda mm: _onset_pianoroll(mm)[0])
provider("onset_pianoroll_positions")(lambda mm: _onset_pianoroll(mm)[1])


# ---------------------------------------------------------------------------
# Reference builders: how the score-side features are computed
# ---------------------------------------------------------------------------
REFERENCE_BUILDERS: Dict[str, Callable[[Any], Any]] = {}


def reference_builder(name: str) -> Callable:
    """Register a ``reference: <name>`` strategy."""

    def decorate(func):
        REFERENCE_BUILDERS[name] = func
        return func

    return decorate


@reference_builder("note_array")
def _reference_note_array(mm):
    """The partitura score note array. The default for every method."""
    return mm.score_part.note_array()


@reference_builder("score_audio")
def _reference_score_audio(mm):
    """Features of a synthesised rendering of the score.

    For audio followers that align the live input against the score *as audio*.
    The same processor then handles the live input, so it is reset afterwards.
    """
    from matchmaker.utils.misc import generate_score_audio

    score_audio = generate_score_audio(mm.score_part, mm.tempo, mm.sample_rate)
    features, _ = mm.processor((score_audio.astype(np.float32), 0.0))
    mm.processor.reset()
    return features


@reference_builder("korzeniowski_score_model")
def _reference_korzeniowski(mm):
    """Harmonic spectral templates for the Korzeniowski particle filter."""
    from matchmaker.features.processor import KorzeniowskiScoreProcessor

    score_processor = KorzeniowskiScoreProcessor(
        sample_rate=_reference_sample_rate(mm),
        n_fft=mm.config.get("n_fft", 4096),
    )
    return score_processor(mm.score_part)


# ---------------------------------------------------------------------------
# Argument resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ArgSpec:
    """One constructor argument, as declared in the YAML."""

    source: str  # "from" | "config" | "value"
    key: Any
    default: Any = None
    has_default: bool = False
    pop: bool = False

    @classmethod
    def parse(cls, name: str, raw: Any, where: str) -> "_ArgSpec":
        """Read one ``args`` entry. Anything not an arg mapping is a literal."""
        if isinstance(raw, dict):
            used = [k for k in _ARG_SOURCES if k in raw]
            if len(used) == 1:
                source = used[0]
                unknown = set(raw) - {source, "default", "pop"}
                if unknown:
                    raise ValueError(
                        f"{where}: argument '{name}' has unknown keys "
                        f"{sorted(unknown)}."
                    )
                spec = cls(
                    source=source,
                    key=raw[source],
                    default=raw.get("default"),
                    has_default="default" in raw,
                    pop=bool(raw.get("pop", False)),
                )
                if source == "from" and spec.key not in PROVIDERS:
                    raise ValueError(
                        f"{where}: argument '{name}' asks for unknown provider "
                        f"'{spec.key}'. Known: {sorted(PROVIDERS)}"
                    )
                return spec
            if len(used) > 1:
                raise ValueError(
                    f"{where}: argument '{name}' sets more than one of "
                    f"{list(_ARG_SOURCES)}: {sorted(used)}."
                )
        return cls(source="value", key=raw)

    def resolve(self, mm) -> Any:
        if self.source == "value":
            return self.key
        if self.source == "from":
            return PROVIDERS[self.key](mm)
        # "config"
        if self.has_default:
            if self.pop:
                return mm.config.pop(self.key, self.default)
            return mm.config.get(self.key, self.default)
        try:
            return mm.config.pop(self.key) if self.pop else mm.config[self.key]
        except KeyError:
            raise KeyError(
                f"'{mm.method}' requires kwargs['{self.key}'], which is not set "
                f"and has no default in the spec."
            ) from None


def _parse_args(raw: Optional[dict], where: str) -> Dict[str, _ArgSpec]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{where}: 'args' must be a mapping, got {type(raw).__name__}."
        )
    return {name: _ArgSpec.parse(name, value, where) for name, value in raw.items()}


def _resolve_args(args: Dict[str, _ArgSpec], mm) -> Dict[str, Any]:
    return {name: spec.resolve(mm) for name, spec in args.items()}


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessorSpec:
    """A feature processor: which class, and what it is constructed with."""

    name: str
    input_type: str
    cls_path: str
    args: Dict[str, _ArgSpec] = field(default_factory=dict)

    @classmethod
    def parse(cls, name: str, input_type: str, raw: dict) -> "ProcessorSpec":
        where = f"processors.{input_type}.{name}"
        if "class" not in raw:
            raise ValueError(f"{where}: missing required key 'class'.")
        unknown = set(raw) - {"class", "args"}
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}.")
        return cls(
            name=name,
            input_type=input_type,
            cls_path=raw["class"],
            args=_parse_args(raw.get("args"), where),
        )

    def build(self, mm, overrides: Optional[Dict[str, Any]] = None):
        kwargs = _resolve_args(self.args, mm)
        if overrides:
            kwargs.update(overrides)
        return load_object(self.cls_path)(**kwargs)


@dataclass(frozen=True)
class MethodSpec:
    """A score follower: which class, arguments, and score-side features."""

    name: str
    input_type: str
    cls_path: str
    args: Dict[str, _ArgSpec] = field(default_factory=dict)
    reference: str = "note_array"
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    processor_args: Dict[str, _ArgSpec] = field(default_factory=dict)
    passthrough: bool = False
    passthrough_exclude: frozenset = frozenset()
    family: Optional[str] = None
    event_based: bool = False
    deterministic: bool = True

    _KEYS = {
        "class",
        "args",
        "reference",
        "default_kwargs",
        "processor_args",
        "config_passthrough",
        "family",
        "event_based",
        "deterministic",
    }

    @classmethod
    def parse(cls, name: str, input_type: str, raw: dict) -> "MethodSpec":
        where = f"methods.{input_type}.{name}"
        if "class" not in raw:
            raise ValueError(f"{where}: missing required key 'class'.")
        unknown = set(raw) - cls._KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}.")

        reference = raw.get("reference", "note_array")
        if reference not in REFERENCE_BUILDERS:
            raise ValueError(
                f"{where}: unknown reference builder '{reference}'. "
                f"Known: {sorted(REFERENCE_BUILDERS)}"
            )

        passthrough_raw = raw.get("config_passthrough", False)
        exclude: frozenset = frozenset()
        if isinstance(passthrough_raw, dict):
            unknown = set(passthrough_raw) - {"exclude"}
            if unknown:
                raise ValueError(
                    f"{where}: unknown config_passthrough keys {sorted(unknown)}."
                )
            exclude = frozenset(passthrough_raw.get("exclude", ()))
            passthrough = True
        else:
            passthrough = bool(passthrough_raw)

        return cls(
            name=name,
            input_type=input_type,
            cls_path=raw["class"],
            args=_parse_args(raw.get("args"), where),
            reference=reference,
            default_kwargs=dict(raw.get("default_kwargs") or {}),
            processor_args=_parse_args(raw.get("processor_args"), where),
            passthrough=passthrough,
            passthrough_exclude=exclude,
            family=raw.get("family"),
            event_based=bool(raw.get("event_based", False)),
            deterministic=bool(raw.get("deterministic", True)),
        )

    def build_reference(self, mm):
        return REFERENCE_BUILDERS[self.reference](mm)

    def build_follower(self, mm):
        kwargs = _resolve_args(self.args, mm)
        if self.passthrough:
            extra = {
                key: value
                for key, value in mm.config.items()
                if key not in self.passthrough_exclude
            }
            clash = sorted(set(extra) & set(kwargs))
            if clash:
                raise ValueError(
                    f"kwargs {clash} clash with arguments '{self.name}' already "
                    f"sets itself; drop them from kwargs."
                )
            kwargs.update(extra)
        return load_object(self.cls_path)(**kwargs)


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------
class Registry:
    """Everything ``methods.yaml`` declares, parsed and validated."""

    def __init__(self, spec: dict, source: Union[str, Path] = "<dict>"):
        self.source = str(source)
        self.version = spec.get("version", 1)

        input_types = spec.get("input_types") or {}
        if not input_types:
            raise ValueError(f"{self.source}: no 'input_types' declared.")

        #: ``{input_type: default method name}``
        self.default_method: Dict[str, str] = {}
        #: ``{input_type: default processor name}``
        self.default_processor: Dict[str, str] = {}
        for input_type, settings in input_types.items():
            self.default_method[input_type] = settings["default_method"]
            self.default_processor[input_type] = settings["default_processor"]

        self.processors: Dict[str, Dict[str, ProcessorSpec]] = {}
        for input_type, entries in (spec.get("processors") or {}).items():
            self._check_input_type(input_type, "processors")
            self.processors[input_type] = {
                name: ProcessorSpec.parse(name, input_type, raw)
                for name, raw in _public(entries)
            }

        self.methods: Dict[str, Dict[str, MethodSpec]] = {}
        for input_type, entries in (spec.get("methods") or {}).items():
            self._check_input_type(input_type, "methods")
            self.methods[input_type] = {
                name: MethodSpec.parse(name, input_type, raw)
                for name, raw in _public(entries)
            }

        for input_type in self.default_method:
            self.processors.setdefault(input_type, {})
            self.methods.setdefault(input_type, {})

        #: ``{input_type: [method names]}``, in declaration order. Mutable:
        #: :func:`matchmaker.matchmaker.register_method` appends to it.
        self.available_methods: Dict[str, List[str]] = {
            input_type: list(methods) for input_type, methods in self.methods.items()
        }
        #: ``{input_type: {method: default kwargs}}``. Mutable, same reason.
        self.default_kwargs: Dict[str, Dict[str, dict]] = {
            input_type: {
                name: dict(spec.default_kwargs) for name, spec in methods.items()
            }
            for input_type, methods in self.methods.items()
        }

        self._validate()

    def _check_input_type(self, input_type: str, section: str) -> None:
        if input_type not in self.default_method:
            raise ValueError(
                f"{self.source}: {section} declares input type '{input_type}', "
                f"which is not in 'input_types'."
            )

    def _validate(self) -> None:
        """Catch spec mistakes at import time rather than at build time."""
        for input_type, default in self.default_method.items():
            if default not in self.methods.get(input_type, {}):
                raise ValueError(
                    f"{self.source}: default method '{default}' for "
                    f"'{input_type}' is not declared."
                )
            default_proc = self.default_processor[input_type]
            if default_proc not in self.processors.get(input_type, {}):
                raise ValueError(
                    f"{self.source}: default processor '{default_proc}' for "
                    f"'{input_type}' is not declared."
                )
        for input_type, methods in self.methods.items():
            known = self.processors.get(input_type, {})
            for name, spec in methods.items():
                declared = spec.default_kwargs.get("processor")
                if declared is not None and declared not in known:
                    raise ValueError(
                        f"methods.{input_type}.{name}: default processor "
                        f"'{declared}' is not declared for {input_type}. "
                        f"Known: {sorted(known)}"
                    )

    # -- lookup ------------------------------------------------------------
    def default_processor_of(self, input_type: str, method: str) -> str:
        """The processor ``method`` runs with unless the user overrides it."""
        spec = self.methods.get(input_type, {}).get(method)
        declared = spec.default_kwargs.get("processor") if spec is not None else None
        return declared or self.default_processor[input_type]

    def method(self, input_type: str, name: str) -> MethodSpec:
        try:
            return self.methods[input_type][name]
        except KeyError:
            raise ValueError(
                f"No {input_type} follower for method '{name}'. "
                f"Available: {sorted(self.methods.get(input_type, {}))}"
            ) from None

    def processor(self, input_type: str, name: str) -> ProcessorSpec:
        try:
            return self.processors[input_type][name]
        except KeyError:
            raise ValueError(
                f"Invalid feature type '{name}' for {input_type}. "
                f"Available: {sorted(self.processors.get(input_type, {}))}"
            ) from None

    def family(self, name: str) -> set:
        """Names of every method tagged with ``family: <name>``."""
        return {
            method
            for methods in self.methods.values()
            for method, spec in methods.items()
            if spec.family == name
        }

    # -- building ----------------------------------------------------------
    def build_processor(self, mm, processor_type: str, method: Optional[str] = None):
        """Instantiate ``processor_type``, with the method's overrides applied.

        ``method`` defaults to the one ``mm`` is running. Pass it explicitly to
        build the processor a *different* method would get — what a follower
        re-registered under a new name needs, so that it sees the same input as
        the built-in it clones.
        """
        spec = self.processor(mm.input_type, processor_type)
        method_spec = self.methods[mm.input_type].get(
            mm.method if method is None else method
        )
        overrides = (
            _resolve_args(method_spec.processor_args, mm)
            if method_spec is not None and method_spec.processor_args
            else None
        )
        return spec.build(mm, overrides)

    def build_follower(self, mm, method: str):
        return self.method(mm.input_type, method).build_follower(mm)

    def build_reference(self, mm, method: str):
        return self.method(mm.input_type, method).build_reference(mm)

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_file(cls, path: Union[str, Path] = SPEC_FILE) -> "Registry":
        with open(path, "r", encoding="utf-8") as spec_file:
            return cls(yaml.load(spec_file, Loader=_SpecLoader), source=str(path))


def _public(entries: Optional[dict]):
    """Spec entries, skipping ``_``-prefixed YAML anchor holders."""
    for name, raw in (entries or {}).items():
        if not name.startswith("_"):
            yield name, raw


#: The registry built from :data:`SPEC_FILE`.
REGISTRY = Registry.from_file()
