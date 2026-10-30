"""
Self-check for --relocate-data: displacements and immediates inside a moved
range are rebased to the copy, and everything else is left alone.

Run: py -3 tools/recomp/test_data_relocations.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from capstone import Cs, CS_ARCH_X86, CS_MODE_32  # noqa: E402

from tools.recomp import disasm  # noqa: E402


def _operands(code):
    cs = Cs(CS_ARCH_X86, CS_MODE_32)
    cs.detail = True
    disasm._init_reg_names(cs)
    insn = next(cs.disasm(code, 0x1000))
    return [disasm._parse_operand(cs, op, insn) for op in insn.operands]


def test_moved_range_is_rebased():
    disasm.DATA_RELOCATIONS[:] = [(0x29CA60, 0xB4, 0x04000000)]
    try:
        # mov al, [eax*2 + 0x29CA62]: the field offset inside the range survives
        assert _operands(bytes.fromhex("8a044562ca2900"))[1].mem_disp == 0x04000002
        # mov edx, 0x29CA6A: a pointer into the table
        assert _operands(bytes.fromhex("ba6aca2900"))[1].imm == 0x0400000A
        # mov [0x29CB14], edi: one past the range is a different variable
        assert _operands(bytes.fromhex("893d14cb2900"))[0].mem_disp == 0x29CB14
        # small immediates and negative displacements are untouched
        assert _operands(bytes.fromhex("b812000000"))[1].imm == 0x12
        assert _operands(bytes.fromhex("8b45fc"))[1].mem_disp == -4
    finally:
        disasm.DATA_RELOCATIONS[:] = []


if __name__ == "__main__":
    test_moved_range_is_rebased()
    print("ok")
