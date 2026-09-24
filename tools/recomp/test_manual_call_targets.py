import hashlib
import json
import os
import tempfile
import types
import unittest

from .__main__ import load_manual_call_targets
from .disasm import Instruction
from .lifter import Lifter
from .translator import BatchTranslator, FunctionTranslator, _fixup_icall_esp_save


TARGET = 0x001E9100
SET_TILE_TARGET = 0x001E4930
OTHER_CALLER = 0x0017A090


def make_recompiler(manual_call_targets, functions, rewrites):
    """Build a recompiler with a stub translator over fixed instructions.

    functions: {start: [Instruction, ...]} standing in for disassembly.
    rewrites: [(caller, callsite, target)] standing in for lifter output.
    """
    recompiler = BatchTranslator.__new__(BatchTranslator)
    recompiler.manual_call_targets = frozenset(manual_call_targets)
    recompiler.manual_call_targets_sha256 = "0" * 64
    recompiler.func_db = {
        start: {"start": f"0x{start:08X}", "end": start + 0x10, "size": 0x10}
        for start in functions
    }
    lifter = types.SimpleNamespace(manual_call_rewrites=list(rewrites))
    translator = types.SimpleNamespace(
        lifter=lifter,
        translated_function_starts=set(functions),
        _recovered_cfg={},
        _read_func_bytes=lambda start, end: b"\x90",
        disasm=types.SimpleNamespace(
            disassemble_function=lambda raw, start, end: functions[start]),
    )
    recompiler.translator = translator
    return recompiler


def direct_call(target=TARGET):
    instruction = Instruction(0x00179A07, 5, "call", hex(target), "e800000000")
    instruction.call_target = target
    return instruction


def call_at(address, target):
    instruction = Instruction(address, 5, "call", hex(target), "e800000000")
    instruction.call_target = target
    return instruction


def jump_at(address, target):
    instruction = Instruction(address, 5, "jmp", hex(target), "e900000000")
    instruction.jump_target = target
    return instruction


class ManualCallTargetLifterTest(unittest.TestCase):
    def test_inventory_decode_does_not_record_a_translation(self):
        info = {"end": 0x1001}
        translator = FunctionTranslator(b"\xc3", {0x1000: info})
        translator._read_func_bytes = lambda start, end: b"\xc3"

        _, blocks = translator.decode_function(0x1000, 0x1001)
        self.assertTrue(blocks)
        self.assertEqual(translator.translated_function_starts, set())
        self.assertIsNotNone(translator.translate_function(0x1000, info))
        self.assertEqual(translator.translated_function_starts, {0x1000})

    def test_selected_direct_call_uses_safe_dispatch(self):
        lifter = Lifter(manual_call_targets={TARGET})

        lifted = lifter.lift_instruction(direct_call())

        self.assertIn(
            "RECOMP_ICALL_SAFE(0x001E9100u, _icall_esp)", lifted[-1])
        self.assertNotIn("sub_001E9100()", "\n".join(lifted))
        self.assertEqual(
            lifter.manual_call_rewrites,
            [(0, 0x00179A07, TARGET)],
        )

    def test_unselected_direct_call_remains_direct(self):
        lifter = Lifter(manual_call_targets={TARGET})

        lifted = lifter.lift_instruction(direct_call(0x001E90D0))

        self.assertIn("RECOMP_ABI_CALL(0x001E90D0u, sub_001E90D0)", lifted[-1])
        self.assertNotIn("RECOMP_ICALL_SAFE", "\n".join(lifted))

    def test_each_selected_target_uses_safe_dispatch(self):
        lifter = Lifter(manual_call_targets={TARGET, SET_TILE_TARGET})

        lifted = lifter.lift_instruction(direct_call(SET_TILE_TARGET))

        self.assertIn(
            "RECOMP_ICALL_SAFE(0x001E4930u, _icall_esp)", lifted[-1])
        self.assertEqual(
            lifter.manual_call_rewrites,
            [(0, 0x00179A07, SET_TILE_TARGET)],
        )

    def test_selected_external_tail_uses_manual_dispatch(self):
        lifter = Lifter(manual_call_targets={TARGET})
        lifter.func_start = 0x000B5570
        lifter.func_end = 0x000B5599
        lifter.current_function_has_ebp = True

        lifted = lifter.lift_instruction(jump_at(0x000B5594, TARGET))

        self.assertIn("RECOMP_ITAIL(0x001E9100u)", lifted[-1])
        self.assertNotIn("sub_001E9100()", lifted[-1])
        self.assertEqual(
            lifter.manual_call_rewrites,
            [(0x000B5570, 0x000B5594, TARGET)],
        )

    def test_argument_pushes_use_pre_argument_saved_esp(self):
        lines = [
            "loc_001799F8:",
            "eax = esp + 0x10;",
            "PUSH32(esp, eax);",
            "PUSH32(esp, 0xA239E8);",
            "PUSH32(esp, 0x40);",
            "PUSH32(esp, ebp);",
            "PUSH32(esp, ebx);",
            "PUSH32(esp, ebp);",
            "g_seh_ebp = ebp; /* publish caller frame */",
            "PUSH32(esp, 0); RECOMP_ICALL_SAFE(0x001E9100u, _icall_esp);",
        ]

        fixed = _fixup_icall_esp_save(lines)

        save_index = fixed.index("{ uint32_t _icall_esp = g_esp;")
        self.assertEqual(fixed[save_index + 1], "PUSH32(esp, eax);")
        self.assertLess(save_index, fixed.index("PUSH32(esp, eax);"))

    def test_ebp_publication_survives_rewrite(self):
        lifter = Lifter(manual_call_targets={TARGET})
        lifter.current_function_has_ebp = True

        lifted = lifter.lift_instruction(direct_call())

        self.assertEqual(
            lifted[0], "g_ebp = ebp; /* frame stays current across calls */")
        self.assertEqual(lifted[1], "g_seh_ebp = ebp;")
        self.assertIn("RECOMP_ICALL_SAFE", lifted[2])

    def test_target_file_requires_matching_sha256(self):
        payload = json.dumps(
            ["0x001E4930", "0x001E9100"]).encode("utf-8")
        with tempfile.NamedTemporaryFile(delete=False) as target_file:
            target_file.write(payload)
            path = target_file.name
        self.addCleanup(os.unlink, path)

        targets, actual_sha256 = load_manual_call_targets(
            path, hashlib.sha256(payload).hexdigest())

        self.assertEqual(targets, frozenset({TARGET, SET_TILE_TARGET}))
        self.assertEqual(actual_sha256, hashlib.sha256(payload).hexdigest())
        with self.assertRaisesRegex(ValueError, "authenticated SHA-256"):
            load_manual_call_targets(path, "0" * 64)


class ManualCallCensusTest(unittest.TestCase):
    def test_exact_census_accepts_matching_receipts(self):
        functions = {
            OTHER_CALLER: [
                call_at(0x0017A109, SET_TILE_TARGET),
                call_at(0x0017A112, SET_TILE_TARGET),
            ],
        }
        recompiler = make_recompiler(
            {SET_TILE_TARGET},
            functions,
            [
                (OTHER_CALLER, 0x0017A109, SET_TILE_TARGET),
                (OTHER_CALLER, 0x0017A112, SET_TILE_TARGET),
            ])

        receipts = recompiler._manual_call_rewrite_receipts()

        self.assertEqual(len(receipts), 2)

    def test_exact_census_includes_selected_tail_jumps(self):
        functions = {
            OTHER_CALLER: [jump_at(0x0017A109, SET_TILE_TARGET)],
        }
        recompiler = make_recompiler(
            {SET_TILE_TARGET},
            functions,
            [(OTHER_CALLER, 0x0017A109, SET_TILE_TARGET)])

        receipts = recompiler._manual_call_rewrite_receipts()

        self.assertEqual(len(receipts), 1)

    def test_selected_call_left_direct_fails_closed(self):
        """A count-only gate passes here; the census must not."""
        functions = {
            OTHER_CALLER: [
                call_at(0x0017A109, SET_TILE_TARGET),
                call_at(0x0017A112, SET_TILE_TARGET),
            ],
        }
        recompiler = make_recompiler(
            {SET_TILE_TARGET},
            functions,
            [(OTHER_CALLER, 0x0017A109, SET_TILE_TARGET)])

        with self.assertRaisesRegex(RuntimeError, "not rewritten"):
            recompiler._manual_call_rewrite_receipts()

    def test_callsite_mismatch_fails_closed(self):
        functions = {OTHER_CALLER: [call_at(0x0017A109, SET_TILE_TARGET)]}
        recompiler = make_recompiler(
            {SET_TILE_TARGET},
            functions,
            [(OTHER_CALLER, 0x0017A11B, SET_TILE_TARGET)])

        with self.assertRaisesRegex(RuntimeError, "census mismatch"):
            recompiler._manual_call_rewrite_receipts()

    def test_unselected_target_rewritten_fails_closed(self):
        functions = {OTHER_CALLER: [call_at(0x0017A109, 0x001E90D0)]}
        recompiler = make_recompiler(
            {SET_TILE_TARGET},
            functions,
            [(OTHER_CALLER, 0x0017A109, 0x001E90D0)])

        with self.assertRaisesRegex(RuntimeError, "unexpected rewrites"):
            recompiler._manual_call_rewrite_receipts()

    def test_unselected_target_remaining_direct_is_accepted(self):
        functions = {
            OTHER_CALLER: [
                call_at(0x0017A109, SET_TILE_TARGET),
                call_at(0x0017A112, 0x001E90D0),
            ],
        }
        recompiler = make_recompiler(
            {SET_TILE_TARGET},
            functions,
            [(OTHER_CALLER, 0x0017A109, SET_TILE_TARGET)])

        receipts = recompiler._manual_call_rewrite_receipts()

        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["target"], "0x001E4930")

    def test_single_function_ignores_other_selected_targets(self):
        """Translating one function must not demand unrelated targets."""
        functions = {
            OTHER_CALLER: [call_at(0x0017A109, SET_TILE_TARGET)],
            0x00179890: [call_at(0x00179A07, TARGET)],
        }
        recompiler = make_recompiler(
            {SET_TILE_TARGET, TARGET},
            functions,
            [(OTHER_CALLER, 0x0017A109, SET_TILE_TARGET)])

        receipts = recompiler._manual_call_rewrite_receipts(
            function_starts={OTHER_CALLER})

        self.assertEqual(len(receipts), 1)
        with self.assertRaises(RuntimeError):
            recompiler._manual_call_rewrite_receipts()


if __name__ == "__main__":
    unittest.main()
