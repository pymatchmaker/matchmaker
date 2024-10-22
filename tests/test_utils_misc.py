import threading
import time
import unittest

import numpy as np

from matchmaker.utils.misc import (
    MatchmakerInvalidOptionError,
    MatchmakerInvalidParameterTypeError,
    MatchmakerMissingParameterError,
    RECVQueue,
    ensure_rng,
)


class TestMatchmakerExceptions(unittest.TestCase):
    def test_invalid_parameter_type_error(self):
        try:
            raise MatchmakerInvalidParameterTypeError(
                parameter_name="param",
                required_parameter_type=(int, float),
                actual_parameter_type=str,
            )
        except MatchmakerInvalidParameterTypeError as e:
            expected_message = "`param` was expected to be <class 'int'>, <class 'float'>, but is <class 'str'>"
            self.assertEqual(str(e), expected_message)

        try:
            raise MatchmakerInvalidParameterTypeError(
                parameter_name="param",
                required_parameter_type=int,
                actual_parameter_type=str,
            )
        except MatchmakerInvalidParameterTypeError as e:
            expected_message = (
                "`param` was expected to be <class 'int'>, but is <class 'str'>"
            )
            self.assertEqual(str(e), expected_message)

    def test_invalid_option_error(self):
        try:
            raise MatchmakerInvalidOptionError(
                parameter_name="option_param",
                valid_options=[
                    "option1",
                    "option2",
                ],
                value="invalid_option",
            )
        except MatchmakerInvalidOptionError as e:
            expected_message = "`option_param` was expected to be in option1, option2, but is invalid_option"
            self.assertEqual(str(e), expected_message)

    def test_missing_parameter_error(self):
        try:
            raise MatchmakerMissingParameterError(parameter_name="missing_param")
        except MatchmakerMissingParameterError as e:
            expected_message = "`missing_param` was not given."
            self.assertEqual(str(e), expected_message)

        try:
            raise MatchmakerMissingParameterError(
                parameter_name=[
                    "param1",
                    "param2",
                ]
            )
        except MatchmakerMissingParameterError as e:
            expected_message = "`param1`, `param2` were not given"
            self.assertEqual(str(e), expected_message)


class TestEnsureRNG(unittest.TestCase):
    def test_seed_as_integer(self):
        rng = ensure_rng(42)
        self.assertIsInstance(rng, np.random.RandomState)
        # Check if the RNG produces consistent results
        rng_expected = np.random.RandomState(42)
        self.assertEqual(rng.randint(0, 100), rng_expected.randint(0, 100))

    def test_seed_as_random_state(self):
        rng_input = np.random.RandomState(123)
        rng = ensure_rng(rng_input)
        self.assertIs(rng, rng_input)

    def test_seed_invalid_type(self):
        with self.assertRaises(ValueError) as context:
            ensure_rng("invalid_seed")
        expected_message = (
            "`seed` should be an integer or an instance of "
            "`np.random.RandomState` but is <class 'str'>"
        )
        self.assertEqual(str(context.exception), expected_message)


class TestRECVQueue(unittest.TestCase):
    def test_recv_method(self):
        queue = RECVQueue()
        test_item = "test_data"
        queue.put(test_item)
        result = queue.recv()
        self.assertEqual(result, test_item)
        self.assertTrue(queue.empty())

    def test_recv_method_with_delay(self):
        queue = RECVQueue()

        def delayed_put(q):
            time.sleep(0.1)
            q.put("delayed_data")

        threading.Thread(target=delayed_put, args=(queue,)).start()
        result = queue.recv()
        self.assertEqual(result, "delayed_data")

    def test_poll_method(self):
        queue = RECVQueue()
        # Queue is empty
        self.assertTrue(queue.poll())
        queue.put("item")
        # Queue is not empty
        self.assertFalse(queue.poll())


if __name__ == "__main__":
    unittest.main()
