"""Tests for the declarative method/processor registry (``methods.yaml``)."""

import inspect
import unittest
import warnings

import yaml

from matchmaker.matchmaker import (
    AVAILABLE_METHODS,
    DEFAULT_KWARGS,
    DEFAULT_METHOD,
    DEFAULT_PROCESSOR,
    OLTW_METHODS,
    PARANGONAR_METHODS,
)
from matchmaker.registry import (
    PROVIDERS,
    _rejected_kwargs,
    REFERENCE_BUILDERS,
    REGISTRY,
    SPEC_FILE,
    Registry,
    load_object,
)

warnings.filterwarnings("ignore", module="partitura")


class TestSpecFile(unittest.TestCase):
    def test_spec_file_ships_with_the_package(self):
        self.assertTrue(SPEC_FILE.is_file(), f"{SPEC_FILE} is missing")

    def test_every_declared_class_is_importable(self):
        for input_type, processors in REGISTRY.processors.items():
            for name, spec in processors.items():
                with self.subTest(processor=f"{input_type}.{name}"):
                    self.assertTrue(callable(load_object(spec.cls_path)))
        for input_type, methods in REGISTRY.methods.items():
            for name, spec in methods.items():
                with self.subTest(method=f"{input_type}.{name}"):
                    self.assertTrue(callable(load_object(spec.cls_path)))

    def test_every_argument_names_a_known_source(self):
        specs = [
            s for group in REGISTRY.processors.values() for s in group.values()
        ] + [s for group in REGISTRY.methods.values() for s in group.values()]
        for spec in specs:
            for arg in spec.args.values():
                if arg.source == "from":
                    self.assertIn(arg.key, PROVIDERS)

    def test_every_reference_builder_is_known(self):
        for methods in REGISTRY.methods.values():
            for name, spec in methods.items():
                with self.subTest(method=name):
                    self.assertIn(spec.reference, REFERENCE_BUILDERS)

    def test_processor_args_are_accepted_by_the_default_processor(self):
        """A method must at least be buildable the way it ships."""
        for input_type, methods in REGISTRY.methods.items():
            for name, spec in methods.items():
                if not spec.processor_args:
                    continue
                processor = REGISTRY.default_processor_of(input_type, name)
                cls = load_object(REGISTRY.processor(input_type, processor).cls_path)
                with self.subTest(method=f"{input_type}.{name}"):
                    self.assertEqual(
                        _rejected_kwargs(cls, spec.processor_args),
                        [],
                        f"methods.{input_type}.{name}.processor_args names "
                        f"arguments its default processor '{processor}' "
                        f"({cls.__name__}) does not accept",
                    )


class TestPublicTables(unittest.TestCase):
    """The module-level tables other packages import must stay live views."""

    def test_tables_are_the_registry_objects(self):
        self.assertIs(AVAILABLE_METHODS, REGISTRY.available_methods)
        self.assertIs(DEFAULT_KWARGS, REGISTRY.default_kwargs)
        self.assertIs(DEFAULT_METHOD, REGISTRY.default_method)
        self.assertIs(DEFAULT_PROCESSOR, REGISTRY.default_processor)

    def test_available_methods_matches_the_spec(self):
        for input_type, methods in REGISTRY.methods.items():
            self.assertEqual(
                AVAILABLE_METHODS[input_type][: len(methods)], list(methods)
            )

    def test_default_method_and_processor_exist(self):
        for input_type, method in DEFAULT_METHOD.items():
            self.assertIn(method, REGISTRY.methods[input_type])
            self.assertIn(
                DEFAULT_PROCESSOR[input_type], REGISTRY.processors[input_type]
            )

    def test_families_are_populated(self):
        self.assertEqual(OLTW_METHODS, {"arzt", "dixon"})
        self.assertEqual(PARANGONAR_METHODS, {"OPTM", "OTM", "SLT_OLTW", "SL_OLTW"})

    def test_obj_tag_resolves_to_the_real_object(self):
        from matchmaker.features.audio import SAMPLE_RATE
        from matchmaker.utils.tempo_models import KalmanTempoModel

        self.assertIs(DEFAULT_KWARGS["midi"]["hmm"]["tempo_model"], KalmanTempoModel)
        self.assertEqual(DEFAULT_KWARGS["audio"]["pfkorz"]["sample_rate"], SAMPLE_RATE)


class TestLoadObject(unittest.TestCase):
    def test_loads_a_dotted_path(self):
        from matchmaker.features.audio import ChromagramProcessor

        self.assertIs(
            load_object("matchmaker.features.audio:ChromagramProcessor"),
            ChromagramProcessor,
        )

    def test_rejects_a_path_without_a_colon(self):
        with self.assertRaises(ValueError):
            load_object("matchmaker.features.audio.ChromagramProcessor")

    def test_reports_a_missing_attribute(self):
        with self.assertRaises(ImportError):
            load_object("matchmaker.features.audio:NoSuchProcessor")


class TestValidation(unittest.TestCase):
    """A malformed spec must fail loudly at load time, not at build time."""

    BASE = """
    version: 1
    input_types:
      midi: {default_method: m, default_processor: p}
    processors:
      midi:
        p: {class: matchmaker.features.midi:PitchProcessor, args: {}}
    methods:
      midi:
        m: {class: matchmaker.prob:PitchHMM, args: {}}
    """

    def _load(self, text):
        return Registry(yaml.safe_load(text), source="<test>")

    def test_the_base_spec_is_valid(self):
        registry = self._load(self.BASE)
        self.assertEqual(registry.available_methods, {"midi": ["m"]})

    def test_unknown_provider_is_rejected(self):
        text = self.BASE.replace(
            "m: {class: matchmaker.prob:PitchHMM, args: {}}",
            "m: {class: matchmaker.prob:PitchHMM, args: {queue: {from: nope}}}",
        )
        with self.assertRaisesRegex(ValueError, "unknown provider"):
            self._load(text)

    def test_unknown_reference_builder_is_rejected(self):
        text = self.BASE.replace(
            "m: {class: matchmaker.prob:PitchHMM, args: {}}",
            "m: {class: matchmaker.prob:PitchHMM, args: {}, reference: nope}",
        )
        with self.assertRaisesRegex(ValueError, "unknown reference builder"):
            self._load(text)

    def test_missing_class_is_rejected(self):
        text = self.BASE.replace(
            "m: {class: matchmaker.prob:PitchHMM, args: {}}", "m: {args: {}}"
        )
        with self.assertRaisesRegex(ValueError, "missing required key 'class'"):
            self._load(text)

    def test_unknown_method_key_is_rejected(self):
        text = self.BASE.replace(
            "m: {class: matchmaker.prob:PitchHMM, args: {}}",
            "m: {class: matchmaker.prob:PitchHMM, args: {}, typo: 1}",
        )
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            self._load(text)

    def test_default_processor_of_a_method_must_exist(self):
        text = self.BASE.replace(
            "m: {class: matchmaker.prob:PitchHMM, args: {}}",
            "m: {class: matchmaker.prob:PitchHMM, args: {}, "
            "default_kwargs: {processor: ghost}}",
        )
        with self.assertRaisesRegex(ValueError, "is not declared"):
            self._load(text)

    def test_default_method_must_exist(self):
        text = self.BASE.replace("default_method: m", "default_method: ghost")
        with self.assertRaisesRegex(ValueError, "default method 'ghost'"):
            self._load(text)

    def test_input_type_must_be_declared(self):
        text = self.BASE + "\n      audio: {}\n"
        with self.assertRaisesRegex(ValueError, "not in 'input_types'"):
            self._load(
                text.replace(
                    "    methods:\n      midi:",
                    "    methods:\n      audio: {}\n      midi:",
                )
            )

    def test_argument_cannot_have_two_sources(self):
        text = self.BASE.replace(
            "m: {class: matchmaker.prob:PitchHMM, args: {}}",
            "m: {class: matchmaker.prob:PitchHMM, "
            "args: {x: {from: tempo, config: tempo}}}",
        )
        with self.assertRaisesRegex(ValueError, "more than one of"):
            self._load(text)


class TestProcessorArgs(unittest.TestCase):
    """``processor_args`` is a requirement, so an incompatible pairing fails."""

    class FakeMM:
        input_type = "midi"
        method = "hmm"

        def __init__(self):
            self.config = {"piano_range": True}

    def test_the_declared_override_is_applied(self):
        self.assertTrue(REGISTRY.build_processor(self.FakeMM(), "pitch").return_pitch_list)

    def test_a_processor_that_cannot_take_it_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            REGISTRY.build_processor(self.FakeMM(), "pianoroll")
        message = str(caught.exception)
        # The reader must learn which method, which argument, which processor,
        # and what to use instead -- the bare TypeError named only the last.
        for expected in ("hmm", "return_pitch_list", "pianoroll", "'pitch'"):
            self.assertIn(expected, message)

    def test_a_method_without_processor_args_is_unaffected(self):
        mm = self.FakeMM()
        mm.method = "pthmm"
        self.assertFalse(REGISTRY.build_processor(mm, "pitch").return_pitch_list)
        REGISTRY.build_processor(mm, "pianoroll")  # no override, no complaint

    def test_the_check_follows_the_method_argument(self):
        """Building another method's processor is checked against that method."""
        mm = self.FakeMM()
        mm.method = "pthmm"
        with self.assertRaisesRegex(ValueError, "hmm"):
            REGISTRY.build_processor(mm, "pianoroll", method="hmm")


class TestRejectedKwargs(unittest.TestCase):
    def test_names_only_what_the_signature_refuses(self):
        class Strict:
            def __init__(self, a=1, b=2):
                pass

        self.assertEqual(_rejected_kwargs(Strict, {"a": 1, "c": 3, "d": 4}), ["c", "d"])

    def test_a_constructor_taking_kwargs_refuses_nothing(self):
        class Open:
            def __init__(self, a=1, **kwargs):
                pass

        self.assertEqual(_rejected_kwargs(Open, {"a": 1, "anything": 2}), [])


class TestLookupErrors(unittest.TestCase):
    def test_unknown_processor_reports_the_alternatives(self):
        with self.assertRaisesRegex(ValueError, "Invalid feature type 'ghost'"):
            REGISTRY.processor("audio", "ghost")

    def test_unknown_method_reports_the_alternatives(self):
        with self.assertRaisesRegex(ValueError, "No audio follower for method"):
            REGISTRY.method("audio", "ghost")


if __name__ == "__main__":
    unittest.main()
