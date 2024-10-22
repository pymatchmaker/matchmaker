#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the utils/stream.py module.
"""
import unittest
from threading import Thread

import numpy as np

from matchmaker.utils.processor import DummyProcessor
from matchmaker.utils.stream import Stream

RNG = np.random.RandomState(1984)


class TestStream(unittest.TestCase):
    def test_init(self):

        # Test initialization
        processor = DummyProcessor()

        for mock in [True, False]:
            stream = Stream(
                processor=processor,
                mock=mock,
            )

            self.assertTrue(isinstance(stream, Stream))

            # Test inheritance
            self.assertTrue(isinstance(stream, Thread))

            stream.start_listening()

            self.assertTrue(stream.listen is True)

            self.assertTrue(stream.mock is mock)

            self.assertTrue(isinstance(stream.init_time, float))

            if stream.mock:
                self.assertTrue(stream.init_time == 0)
            else:
                self.assertTrue(stream.init_time != 0)

            # Test that methods raise the correct errors

            data = RNG.rand(100, 71)
            self.assertRaises(NotImplementedError, stream._process_frame, data)

            self.assertRaises(NotImplementedError, stream.mock_stream)

            self.assertRaises(NotImplementedError, stream.run)

            self.assertRaises(NotImplementedError, stream.stop)

            # Use assertRaises as a context manager to check if entering the context raises an exception
            with self.assertRaises(NotImplementedError):
                with stream as st:
                    pass

            stream.stop_listening()


if __name__ == "__main__":
    unittest.main()
