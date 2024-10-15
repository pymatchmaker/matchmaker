#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the utils.processor module.
"""
import unittest

import numpy as np

from matchmaker.utils.processor import Processor, ProcessorWrapper

RNG = np.random.RandomState(1984)


class TestProcessor(unittest.TestCase):

    def test_raise_not_implemented(self):

        data = RNG.rand(100, 7)
        processor = Processor()
        processor.reset()
        self.assertRaises(NotImplementedError, processor, data)


class TestProcessorWrapper(unittest.TestCase):

    def test_init(self):

        func = lambda x: 2 * x
        processor = ProcessorWrapper(func=func)
        data = RNG.rand(100, 7)
        proc_output = processor(data)
        expected_output = func(data)
        self.assertTrue(np.all(proc_output == expected_output))
