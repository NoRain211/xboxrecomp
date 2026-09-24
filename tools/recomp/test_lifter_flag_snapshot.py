import unittest

from .disasm import BasicBlock, Instruction, Operand
from .lifter import Lifter, lift_basic_block


class FlagSnapshotLifterTest(unittest.TestCase):
    """Deferred conditions must not re-read a clobbered flag operand.

    CMP and TEST capture their operands at the point where they set EFLAGS.
    A move between the compare and its consumer preserves EFLAGS but changes
    a register value, so the consumer must use those captured operands.
    """

    def _setl_after_clobber(self):
        """cmp eax, 1 / mov eax, [0x2A7AA4] / setl cl (sub_000F39E0)."""
        cmp = Instruction(0x000F39E8, 3, "cmp", "eax, 1", "83f801")
        cmp.operands = [
            Operand(type="reg", reg="eax"),
            Operand(type="imm", imm=1),
        ]
        mov = Instruction(0x000F39EB, 5, "mov", "eax, dword ptr [0x2a7aa4]",
                          "a1a47a2a00")
        mov.operands = [
            Operand(type="reg", reg="eax"),
            Operand(type="mem", mem_disp=0x2A7AA4, mem_size=4),
        ]
        setl = Instruction(0x000F39F0, 3, "setl", "cl", "0f9cc1")
        setl.operands = [Operand(type="reg", reg="cl")]
        return [cmp, mov, setl]

    def test_setcc_reads_snapshot_not_clobbered_operand(self):
        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(start=0x000F39E8,
                       instructions=self._setl_after_clobber()))
        generated = "\n".join(lifted)

        self.assertIn("_fa = (uint32_t)(eax)", generated)

        setcc = [ln for ln in lifted if "setl" in ln]
        self.assertEqual(len(setcc), 1)
        self.assertIn("CMP_L(_fas, _fbs)", setcc[0])
        # The whole point: the condition must not name the post-mov register.
        self.assertNotIn("CMP_L(eax,", setcc[0])

    def test_snapshot_precedes_the_clobbering_move(self):
        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(start=0x000F39E8,
                       instructions=self._setl_after_clobber()))

        snap = next(i for i, ln in enumerate(lifted)
                    if "_fa = (uint32_t)(eax)" in ln)
        clobber = next(i for i, ln in enumerate(lifted)
                       if "0x2A7AA4" in ln or "0x2a7aa4" in ln)
        self.assertLess(snap, clobber)

    def test_no_snapshot_when_operand_survives(self):
        """An untouched operand still lifts to a plain re-read."""
        cmp = Instruction(0x1000, 3, "cmp", "eax, 1", "83f801")
        cmp.operands = [
            Operand(type="reg", reg="eax"),
            Operand(type="imm", imm=1),
        ]
        mov = Instruction(0x1003, 5, "mov", "edx, dword ptr [0x2a7aa4]",
                          "8b15a47a2a00")
        mov.operands = [
            Operand(type="reg", reg="edx"),
            Operand(type="mem", mem_disp=0x2A7AA4, mem_size=4),
        ]
        setl = Instruction(0x1008, 3, "setl", "cl", "0f9cc1")
        setl.operands = [Operand(type="reg", reg="cl")]

        lifted, _ = lift_basic_block(
            Lifter(), BasicBlock(start=0x1000, instructions=[cmp, mov, setl]))
        generated = "\n".join(lifted)

        self.assertNotIn("_flagsnap", generated)
        setcc = [ln for ln in lifted if "setl" in ln]
        self.assertEqual(len(setcc), 1)
        self.assertIn("CMP_L(_fas, _fbs)", setcc[0])

    def test_cmovcc_reads_snapshot(self):
        """CMOVcc defers the same way SETcc does."""
        cmp = Instruction(0x2000, 2, "cmp", "esi, edi", "39fe")
        cmp.operands = [
            Operand(type="reg", reg="esi"),
            Operand(type="reg", reg="edi"),
        ]
        mov = Instruction(0x2002, 2, "mov", "esi, ebx", "89de")
        mov.operands = [
            Operand(type="reg", reg="esi"),
            Operand(type="reg", reg="ebx"),
        ]
        cmovl = Instruction(0x2004, 3, "cmovl", "eax, edx", "0f4cc2")
        cmovl.operands = [
            Operand(type="reg", reg="eax"),
            Operand(type="reg", reg="edx"),
        ]

        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(start=0x2000, instructions=[cmp, mov, cmovl]))
        generated = "\n".join(lifted)

        self.assertIn("_fa = (uint32_t)(esi)", generated)
        cmov = [ln for ln in lifted if "cmov" in ln]
        self.assertEqual(len(cmov), 1)
        self.assertIn("CMP_L(_fas, _fbs)", cmov[0])

    def test_non_adjacent_jcc_reads_snapshot(self):
        """A jcc separated from its cmp defers exactly like SETcc does.

        sub_000F4330, the screen-1 update:

            000F4354  cmp eax, 1          result of the phase query
            000F4357  mov eax, [esp+4]    eax reused for the phase word
            000F435B  jne loc_000F4387

        try_match_cmp_jcc only pairs an *adjacent* cmp/jcc, so this jump
        rebuilt "eax != 1" from the post-mov eax and took the wrong arm:
        screen 1 dispatched on the phase word instead of the query result.
        """
        cmp = Instruction(0x000F4354, 3, "cmp", "eax, 1", "83f801")
        cmp.operands = [
            Operand(type="reg", reg="eax"),
            Operand(type="imm", imm=1),
        ]
        mov = Instruction(0x000F4357, 4, "mov", "eax, dword ptr [esp + 4]",
                          "8b442404")
        mov.operands = [
            Operand(type="reg", reg="eax"),
            Operand(type="mem", mem_base="esp", mem_disp=4, mem_size=4),
        ]
        jne = Instruction(0x000F435B, 2, "jne", "0xf4387", "752a")
        jne.operands = [Operand(type="imm", imm=0x000F4387)]
        jne.jump_target = 0x000F4387

        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(start=0x000F4354, instructions=[cmp, mov, jne]))
        generated = "\n".join(lifted)

        self.assertIn("_fa = (uint32_t)(eax)", generated)

        jcc = [ln for ln in lifted if "jne:" in ln]
        self.assertEqual(len(jcc), 1)
        self.assertIn("CMP_NE(_fa, _fb)", jcc[0])
        # The whole point: the jump must not test the reloaded register.
        self.assertNotIn("CMP_NE(eax,", jcc[0])

    def test_test_al_jge_keeps_8_bit_signed_snapshot(self):
        test = Instruction(0x0015B45E, 2, "test", "al, al", "84c0")
        test.operands = [
            Operand(type="reg", reg="al"),
            Operand(type="reg", reg="al"),
        ]
        mov = Instruction(
            0x0015B460, 5, "mov", "ecx, 0x16", "b916000000")
        mov.operands = [
            Operand(type="reg", reg="ecx"),
            Operand(type="imm", imm=0x16),
        ]
        lea = Instruction(
            0x0015B465, 6, "lea", "edi, dword ptr [ebp + 0x9d5240]",
            "8dbd40529d00")
        lea.operands = [
            Operand(type="reg", reg="edi"),
            Operand(
                type="mem", mem_base="ebp", mem_disp=0x9D5240, mem_size=4),
        ]
        jge = Instruction(0x0015B46B, 2, "jge", "0x15b475", "7d08")
        jge.operands = [Operand(type="imm", imm=0x0015B475)]
        jge.jump_target = 0x0015B475

        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(
                start=0x0015B45E,
                instructions=[test, mov, lea, jge],
            ),
        )
        generated = "\n".join(lifted)

        self.assertIn("_fas = (int32_t)(int8_t)(_fa)", generated)
        self.assertIn("CMP_GE((int8_t)((_fas) & (_fbs)), 0)", generated)

    def test_adjacent_jcc_needs_no_snapshot(self):
        """The adjacent pair is already exact, so nothing is captured."""
        cmp = Instruction(0x3000, 3, "cmp", "eax, 1", "83f801")
        cmp.operands = [
            Operand(type="reg", reg="eax"),
            Operand(type="imm", imm=1),
        ]
        jne = Instruction(0x3003, 2, "jne", "0x3020", "751b")
        jne.operands = [Operand(type="imm", imm=0x3020)]
        jne.jump_target = 0x3020

        lifted, _ = lift_basic_block(
            Lifter(), BasicBlock(start=0x3000, instructions=[cmp, jne]))
        generated = "\n".join(lifted)

        self.assertNotIn("_flagsnap", generated)
        self.assertIn("CMP_NE(_fa, _fb)", generated)

class ImplicitClobberSnapshotTest(unittest.TestCase):
    """A clobber the operand list never names is still a clobber.

    _written_regs read operand zero, so an instruction that overwrites a
    register implicitly reported nothing and the deferred condition re-read
    a value that had already been replaced.
    """

    def test_cdq_clobber_is_snapshotted(self):
        """cmp edx, 3 / cdq / jne -- CDQ rewrites EDX from EAX's sign.

        Observed at recomp_0011.c:9831 in the shipped snapshot, where the
        jump compared the freshly sign-extended EDX against 3 instead of the
        value the CMP actually tested.
        """
        cmp = Instruction(0x00198AE1, 3, "cmp", "edx, 3", "83fa03")
        cmp.operands = [
            Operand(type="reg", reg="edx"),
            Operand(type="imm", imm=3),
        ]
        cdq = Instruction(0x00198AE4, 1, "cdq", "", "99")
        cdq.operands = []
        jne = Instruction(0x00198AE5, 2, "jne", "0x198b37", "7550")
        jne.operands = [Operand(type="imm", imm=0x00198B37)]
        jne.jump_target = 0x00198B37

        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(start=0x00198AE1, instructions=[cmp, cdq, jne]))
        generated = "\n".join(lifted)

        self.assertIn("_fa = (uint32_t)(edx)", generated)
        jcc = [ln for ln in lifted if "jne:" in ln]
        self.assertEqual(len(jcc), 1)
        self.assertIn("CMP_NE(_fa, _fb)", jcc[0])
        self.assertNotIn("CMP_NE(edx,", jcc[0])

    def test_cdq_not_snapshotted_when_edx_is_not_compared(self):
        """CDQ only matters when the condition actually reads EDX."""
        cmp = Instruction(0x1000, 2, "cmp", "esi, edi", "39fe")
        cmp.operands = [
            Operand(type="reg", reg="esi"),
            Operand(type="reg", reg="edi"),
        ]
        cdq = Instruction(0x1002, 1, "cdq", "", "99")
        cdq.operands = []
        jne = Instruction(0x1003, 2, "jne", "0x1020", "751b")
        jne.operands = [Operand(type="imm", imm=0x1020)]
        jne.jump_target = 0x1020

        lifted, _ = lift_basic_block(
            Lifter(), BasicBlock(start=0x1000, instructions=[cmp, cdq, jne]))
        generated = "\n".join(lifted)

        self.assertNotIn("_flagsnap", generated)
        self.assertIn("CMP_NE(_fa, _fb)", generated)

    def test_xchg_clobbers_both_operands(self):
        """XCHG replaces both registers, not just the destination."""
        cmp = Instruction(0x2000, 2, "cmp", "eax, 1", "83f801")
        cmp.operands = [
            Operand(type="reg", reg="eax"),
            Operand(type="imm", imm=1),
        ]
        xchg = Instruction(0x2002, 2, "xchg", "ecx, eax", "91")
        xchg.operands = [
            Operand(type="reg", reg="ecx"),
            Operand(type="reg", reg="eax"),
        ]
        setl = Instruction(0x2004, 3, "setl", "dl", "0f9cc2")
        setl.operands = [Operand(type="reg", reg="dl")]

        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(start=0x2000, instructions=[cmp, xchg, setl]))
        generated = "\n".join(lifted)

        self.assertIn("_fa = (uint32_t)(eax)", generated)
        setcc = [ln for ln in lifted if "setl" in ln]
        self.assertIn("CMP_L(_fas, _fbs)", setcc[0])


class X87InterleavedFlagTest(unittest.TestCase):
    """x87 arithmetic preserves EFLAGS, so a deferred condition survives it.

    The FPU reports comparisons through its own status word; only the
    FCOMI/FUCOMI family and SAHF reach EFLAGS. Treating ordinary x87 work as
    flag-destroying aborted the clobber scan, which suppressed the snapshot
    while the consumer still emitted the comparison.
    """

    def _cmp_fpu_jcc(self, clobber):
        cmp = Instruction(0x3000, 2, "cmp", "esi, eax", "39c6")
        cmp.operands = [
            Operand(type="reg", reg="esi"),
            Operand(type="reg", reg="eax"),
        ]
        fstp = Instruction(0x3002, 3, "fstp", "dword ptr [ecx]", "d919")
        fstp.operands = [Operand(type="mem", mem_base="ecx", mem_size=4)]
        jb = Instruction(0x3008, 2, "jb", "0x3040", "7236")
        jb.operands = [Operand(type="imm", imm=0x3040)]
        jb.jump_target = 0x3040
        return [cmp, fstp, clobber, jb] if clobber else [cmp, fstp, jb]

    def test_clobber_across_x87_is_snapshotted(self):
        """cmp esi, eax / fstp / mov eax, ... / jb (recomp_0001.c:47810)."""
        mov = Instruction(0x3005, 3, "mov", "eax, edx", "89d0")
        mov.operands = [
            Operand(type="reg", reg="eax"),
            Operand(type="reg", reg="edx"),
        ]
        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(start=0x3000, instructions=self._cmp_fpu_jcc(mov)))
        generated = "\n".join(lifted)

        self.assertIn("_fa = (uint32_t)(esi)", generated)
        jcc = [ln for ln in lifted if "jb:" in ln]
        self.assertEqual(len(jcc), 1)
        self.assertIn("CMP_B(_fa, _fb)", jcc[0])

    def test_x87_alone_needs_no_snapshot(self):
        """Float work that touches no compared register changes nothing."""
        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(start=0x3000, instructions=self._cmp_fpu_jcc(None)))
        generated = "\n".join(lifted)

        self.assertNotIn("_flagsnap", generated)
        self.assertIn("CMP_B(_fa, _fb)", generated)

    def test_fcomi_still_sets_flags(self):
        """FCOMI writes EFLAGS, so it must replace the pending comparison."""
        cmp = Instruction(0x4000, 2, "cmp", "esi, eax", "39c6")
        cmp.operands = [
            Operand(type="reg", reg="esi"),
            Operand(type="reg", reg="eax"),
        ]
        fcomi = Instruction(0x4002, 2, "fcomi", "st(1)", "dbf1")
        fcomi.operands = [Operand(type="reg", reg="st(1)")]
        jb = Instruction(0x4004, 2, "jb", "0x4040", "723a")
        jb.operands = [Operand(type="imm", imm=0x4040)]
        jb.jump_target = 0x4040

        lifted, _ = lift_basic_block(
            Lifter(),
            BasicBlock(start=0x4000, instructions=[cmp, fcomi, jb]))
        generated = "\n".join(lifted)

        # The jump belongs to the FPU compare, not the integer CMP.
        self.assertNotIn("CMP_B(esi, eax)", generated)


if __name__ == "__main__":
    unittest.main()
