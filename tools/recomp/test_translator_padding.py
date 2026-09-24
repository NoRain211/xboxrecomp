"""Regression checks for alignment padding recognized by CFG recovery."""
import unittest

from .disasm import Disassembler
from .translator import FunctionTranslator


class TranslatorPaddingTest(unittest.TestCase):
    def test_register_self_move_is_padding(self):
        insn = Disassembler().disassemble_function(bytes.fromhex("8bff"), 0, 2)[0]
        self.assertTrue(FunctionTranslator._is_multi_byte_nop(insn))

    def test_register_move_with_different_source_is_not_padding(self):
        insn = Disassembler().disassemble_function(bytes.fromhex("8bfe"), 0, 2)[0]
        self.assertFalse(FunctionTranslator._is_multi_byte_nop(insn))


if __name__ == "__main__":
    unittest.main()
