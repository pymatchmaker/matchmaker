#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the utils.processor module.
"""
import unittest

import numpy as np

from matchmaker.utils.processor import DummyProcessor, Processor, ProcessorWrapper

RNG = np.random.RandomState(1984)


class TestProcessor(unittest.TestCase):
    """
    Tests for `Processor` class
    """

    def test_raise_not_implemented(self):
        data = RNG.rand(100, 7)
        processor = Processor()
        processor.reset()
        self.assertRaises(NotImplementedError, processor, data)


class TestProcessorWrapper(unittest.TestCase):
    """
    Tests for `ProcessorWrapper` class
    """

    def test_init(self):
        func = lambda x: 2 * x
        processor = ProcessorWrapper(func=func)
        data = RNG.rand(100, 7)
        proc_output = processor(data)
        expected_output = func(data)
        self.assertTrue(np.all(proc_output == expected_output))


class TestDummyProcessor(unittest.TestCase):
    """
    Tests for `DummyProcessor` class
    """

    def test_init(self):
        processor = DummyProcessor()
        data = RNG.rand(100, 7)
        proc_output = processor(data)
        self.assertTrue(np.all(proc_output == data))
