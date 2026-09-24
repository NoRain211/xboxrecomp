"""Explicit x87 destination registers must survive translation."""
import unittest

from .disasm import Disassembler, Instruction, Operand
from .lifter import Lifter


class FpuDestinationTest(unittest.TestCase):
    def test_decoded_reverse_destinations(self):
        for raw, dest, operator in [('dce1', 'fp_st1()', '-'), ('dcf6', 'fp_st(6)', '/')]:
            instructions = Disassembler().disassemble_function(bytes.fromhex(raw), 0x1000, 0x1002)
            self.assertEqual(len(instructions), 1)
            text = Lifter().lift_instruction(instructions[0])[0]
            self.assertIn(f'{dest} = fp_st(0) {operator} {dest};', text)

    def test_two_register_arithmetic(self):
        for mnemonic, operator in [('fadd', '+'), ('fsub', '-'), ('fmul', '*'),
                                   ('fdiv', '/'), ('fsubr', '-'), ('fdivr', '/')]:
            for dest, source in [(1, 0), (6, 0), (0, 3)]:
                with self.subTest(mnemonic=mnemonic, dest=dest, source=source):
                    insn = Instruction(0, 2, mnemonic, f'st({dest}), st({source})', '')
                    insn.operands = [Operand(type='reg', reg=f'st({i})') for i in (dest, source)]
                    text = Lifter().lift_instruction(insn)[0].replace('fp_st1()', 'fp_st(1)').replace('fp_top()', 'fp_st(0)')
                    expected = (f'fp_st({dest}) = fp_st({source}) {operator} fp_st({dest});'
                                if mnemonic.endswith('r') else
                                f'fp_st({dest}) {operator}= fp_st({source});')
                    self.assertIn(expected, text)
                    self.assertNotIn('fp_pop', text)
