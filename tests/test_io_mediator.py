#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for the matchmaker.io.mediator module
"""
import unittest
from collections import namedtuple
from matchmaker.io.mediator import ThreadMediator, CeusMediator


class TestThreadMediator(unittest.TestCase):
    def setUp(self):
        self.mediator = ThreadMediator()

    def test_init(self):
        """Test that the mediator initializes correctly."""
        self.assertTrue(self.mediator.is_empty())
        self.assertEqual(len(self.mediator._comms_buffer), 0)
        self.assertEqual(self.mediator.mediator_type, "default")

    def test_put_and_get_message(self):
        """Test putting and getting messages from the buffer."""
        Message = namedtuple("Message", ["type", "value"])
        msg1 = Message(type="note_on", value=60)
        msg2 = Message(type="note_off", value=60)

        self.mediator.put_message(msg1)
        self.mediator.put_message(msg2)

        self.assertFalse(self.mediator.is_empty())
        self.assertEqual(len(self.mediator._comms_buffer), 2)

        retrieved_msg1 = self.mediator.get_message()
        retrieved_msg2 = self.mediator.get_message()

        self.assertEqual(retrieved_msg1, msg1)
        self.assertEqual(retrieved_msg2, msg2)
        self.assertTrue(self.mediator.is_empty())

    def test_get_message_empty_buffer(self):
        """Test that getting a message from an empty buffer raises IndexError."""
        with self.assertRaises(IndexError):
            self.mediator.get_message()

    def test_comms_buffer_maxlen(self):
        """Test that the buffer respects its maximum length."""
        Message = namedtuple("Message", ["type", "value"])
        for i in range(250):
            self.mediator.put_message(Message(type="note_on", value=i))

        self.assertEqual(len(self.mediator._comms_buffer), 200)
        self.assertEqual(self.mediator._comms_buffer[0].value, 50)


class TestCeusMediator(unittest.TestCase):
    def setUp(self):
        self.ceus_mediator = CeusMediator()

    def test_init(self):
        """Test that the Ceus mediator initializes correctly."""
        self.assertEqual(len(self.ceus_mediator._ceus_filter), 0)
        self.assertEqual(self.ceus_mediator.mediator_type, "ceus")

    def test_filter_append_and_check(self):
        """Test appending pitches to the filter and checking them."""
        self.ceus_mediator.filter_append_pitch(60)
        self.assertTrue(self.ceus_mediator.filter_check(60))
        self.assertFalse(self.ceus_mediator.filter_check(60))

    def test_filter_check_without_deletion(self):
        """Test checking pitches without deleting them from the filter."""
        self.ceus_mediator.filter_append_pitch(61)
        self.assertTrue(self.ceus_mediator.filter_check(61, delete_entry=False))
        self.assertTrue(self.ceus_mediator.filter_check(61))

    def test_filter_remove_pitch(self):
        """Test removing pitches from the filter."""
        self.ceus_mediator.filter_append_pitch(62)
        self.ceus_mediator.filter_remove_pitch(62)
        self.assertFalse(self.ceus_mediator.filter_check(62))
        with self.assertRaises(ValueError):
            self.ceus_mediator.filter_remove_pitch(62)

    def test_ceus_filter_maxlen(self):
        """Test that the Ceus filter respects its maximum length."""
        for i in range(15):
            self.ceus_mediator.filter_append_pitch(i)
        self.assertEqual(len(self.ceus_mediator._ceus_filter), 10)
        self.assertFalse(0 in self.ceus_mediator._ceus_filter)
        self.assertTrue(5 in self.ceus_mediator._ceus_filter)


if __name__ == "__main__":
    unittest.main()
