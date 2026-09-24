"""Tests for same-start recovery extensions in adopt_external_functions.

A detected function can end at the first target of its own computed jump,
which truncates the body and leaves the interior labels unreachable. An
explicit recovery entry repeating that start may push the end outward, but
only when the requested extent is proven. These tests use synthetic bytes
assembled here, never the private XBE.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from . import config
from .config import va_to_file_offset
from .translator import BatchTranslator, FunctionTranslator


BASE = 0x00020000
NEXT = 0x00020100
UNRELATED = 0x00020400
# Disjoint adoption requires a following detected function to bound it.
SENTINEL = 0x00020500


def build_xbe(body_at):
    """Place synthetic code at real VAs inside .text."""
    size = va_to_file_offset(UNRELATED + 0x100)
    image = bytearray(b"\xCC" * size)
    for va, payload in body_at.items():
        offset = va_to_file_offset(va)
        image[offset:offset + len(payload)] = payload
    return bytes(image)


# A truncated dispatch-shaped body:
#   0x20000: 31 C0        xor eax, eax
#   0x20002: 85 C0        test eax, eax
#   0x20004: 74 03        je   0x20009
#   0x20006: C2 08 00     ret 8          <- detection stops here
#   0x20009: 40           inc eax
#   0x2000A: C2 08 00     ret 8
TRUNCATED_BODY = bytes.fromhex("31c085c07403c2080040c20800")
TRUNCATED_START = BASE
DETECTED_END = BASE + 9
FULL_END = BASE + len(TRUNCATED_BODY)

# A standalone body for the disjoint-adoption path: xor eax, eax; ret 8.
DISJOINT_BODY = bytes.fromhex("31c0c20800")
DISJOINT_END = UNRELATED + len(DISJOINT_BODY)

# A split computed-jump body whose two-entry table starts exactly at the
# owned end. The coalescer must read the table through the following gap while
# retaining BODY_END as the function extent.
TABLE_BODY_START = BASE + 0x20
TABLE_CASE = TABLE_BODY_START + 9
TABLE_BODY_END = TABLE_BODY_START + 13
TABLE_ADDRESS = TABLE_BODY_END
TABLE_BODY = (
    bytes.fromhex("31c0ff2485")
    + TABLE_ADDRESS.to_bytes(4, "little")
    + bytes.fromhex("40c34bc3")
    + TABLE_CASE.to_bytes(4, "little")
    + (TABLE_CASE + 2).to_bytes(4, "little"))

BRIDGE_BODY_START = BASE + 0x40
BRIDGE_GAP = BRIDGE_BODY_START + 2
BRIDGE_CASE = BRIDGE_BODY_START + 4
BRIDGE_BODY_END = BRIDGE_BODY_START + 5
# jmp case; two-byte nop; ret
BRIDGE_BODY = bytes.fromhex("eb026690c3")

# The same shape padded with the ``lea ebx, [ebx]`` alignment form the retail
# XBE actually uses instead of an explicit nop.
LEA_BODY_START = BASE + 0x60
LEA_GAP = LEA_BODY_START + 2
LEA_CASE = LEA_BODY_START + 4
LEA_BODY_END = LEA_BODY_START + 5
# jmp case; lea ebx, [ebx] (two-byte form); ret
LEA_BODY = bytes.fromhex("eb028d1bc3")

# A bridge whose gap is a real instruction rather than padding: the coalescer
# must refuse it even though the extent still tiles.
LIVE_BODY_START = BASE + 0x80
LIVE_GAP = LIVE_BODY_START + 2
LIVE_CASE = LIVE_BODY_START + 4
LIVE_BODY_END = LIVE_BODY_START + 5
# jmp case; xor eax, eax; ret
LIVE_BODY = bytes.fromhex("eb0231c0c3")

# Two separate pads, so the first bridge is genuine padding that still stops
# short of the only named false start.
UNNAMED_BODY_START = BASE + 0xA0
UNNAMED_GAP = UNNAMED_BODY_START + 2
UNNAMED_CASE = UNNAMED_BODY_START + 6
UNNAMED_BODY_END = UNNAMED_BODY_START + 7
# jmp case; nop; nop; ret
UNNAMED_BODY = bytes.fromhex("eb0466906690c3")

# Padding that lands mid-body rather than on a named false start, which is the
# shape the retail XBE actually uses: MSVC aligns an interior branch target,
# and the instruction after the pad is ordinary code no entry names. Nothing
# may need to be declared for this to coalesce.
AUTOPAD_BODY_START = BASE + 0xC0
AUTOPAD_GAP = AUTOPAD_BODY_START + 2
AUTOPAD_RESUME = AUTOPAD_BODY_START + 4
AUTOPAD_CASE = AUTOPAD_BODY_START + 6
AUTOPAD_BODY_END = AUTOPAD_BODY_START + 7
# jmp resume; lea ebx, [ebx]; xor eax, eax; ret
AUTOPAD_BODY = bytes.fromhex("eb028d1b31c0c3")

# The same shape with real code in the gap instead of padding. The decode
# still leaves a hole, but nothing may close it.
AUTOLIVE_BODY_START = BASE + 0xE0
AUTOLIVE_GAP = AUTOLIVE_BODY_START + 2
AUTOLIVE_BODY_END = AUTOLIVE_BODY_START + 7
# jmp +2; xor eax, eax; xor eax, eax; ret
AUTOLIVE_BODY = bytes.fromhex("eb0231c031c0c3")

# An extension with the retail eight-byte alignment sequence: a seven-byte
# lea esp, [esp] followed by a one-byte nop.
EXT_AUTOPAD_BODY_START = BASE + 0xF0
EXT_AUTOPAD_GAP = EXT_AUTOPAD_BODY_START + 2
EXT_AUTOPAD_RESUME = EXT_AUTOPAD_BODY_START + 10
EXT_AUTOPAD_BODY_END = EXT_AUTOPAD_BODY_START + 11
EXT_AUTOPAD_BODY = bytes.fromhex("eb088da4240000000090c3")


def _thunk(start, end):
    """Build a weak detected entry for one span of a split body."""
    return {
        "_addr": start,
        "start": f"0x{start:08X}",
        "end": end,
        "size": end - start,
        "name": f"sub_{start:08X}",
        "section": ".text",
        "num_instructions": 1,
        "detection_method": "seed_vtable_thunk",
        "has_prologue": False,
        "calls_to": [],
        "called_by": [],
    }


def make_translator(extra_functions=None):
    xbe = build_xbe({
        TRUNCATED_START: TRUNCATED_BODY,
        TABLE_BODY_START: TABLE_BODY,
        BRIDGE_BODY_START: BRIDGE_BODY,
        LEA_BODY_START: LEA_BODY,
        LIVE_BODY_START: LIVE_BODY,
        UNNAMED_BODY_START: UNNAMED_BODY,
        AUTOPAD_BODY_START: AUTOPAD_BODY,
        AUTOLIVE_BODY_START: AUTOLIVE_BODY,
        EXT_AUTOPAD_BODY_START: EXT_AUTOPAD_BODY,
        UNRELATED: DISJOINT_BODY,
    })
    func_db = {
        TRUNCATED_START: {
            "_addr": TRUNCATED_START,
            "start": f"0x{TRUNCATED_START:08X}",
            "end": DETECTED_END,
            "size": DETECTED_END - TRUNCATED_START,
            "name": "sub_00020000",
            "section": ".text",
            "num_instructions": 4,
            "detection_method": "seed_vtable_thunk",
            "has_prologue": False,
            "calls_to": [],
            "called_by": [],
        },
        NEXT: {
            "_addr": NEXT,
            "start": f"0x{NEXT:08X}",
            "end": NEXT + 0x10,
            "size": 0x10,
            "name": "sub_00020100",
            "section": ".text",
            "num_instructions": 1,
            "detection_method": "seed_vtable_thunk",
            "has_prologue": False,
            "calls_to": [],
            "called_by": [],
        },
        SENTINEL: {
            "_addr": SENTINEL,
            "start": f"0x{SENTINEL:08X}",
            "end": SENTINEL + 0x10,
            "size": 0x10,
            "name": "sub_00020500",
            "section": ".text",
            "num_instructions": 1,
            "detection_method": "seed_vtable_thunk",
            "has_prologue": False,
            "calls_to": [],
            "called_by": [],
        },
    }
    for start, info in (extra_functions or {}).items():
        func_db[start] = info
    return FunctionTranslator(xbe, func_db), func_db


class RecoveryExtensionTest(unittest.TestCase):
    def setUp(self):
        # Synthetic addresses must not depend on a previous test's XBE layout.
        sections = patch.object(config, "_SECTIONS", [
            config.Section(".text", BASE, 0x600, 0, 0x600, True),
        ])
        sections.start()
        self.addCleanup(sections.stop)

    def test_batch_translator_layers_recovery_files(self):
        """Repeated recovery inputs preserve the baseline and add the delta."""
        xbe = build_xbe({
            TRUNCATED_START: TRUNCATED_BODY,
            UNRELATED: DISJOINT_BODY,
        })
        functions = [
            {
                "start": f"0x{TRUNCATED_START:08X}",
                "end": f"0x{DETECTED_END:08X}",
                "size": DETECTED_END - TRUNCATED_START,
                "name": "sub_00020000",
                "section": ".text",
                "num_instructions": 4,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
            {
                "start": f"0x{NEXT:08X}",
                "end": f"0x{NEXT + 0x10:08X}",
                "size": 0x10,
                "name": "sub_00020100",
                "section": ".text",
                "num_instructions": 1,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
            {
                "start": f"0x{SENTINEL:08X}",
                "end": f"0x{SENTINEL + 0x10:08X}",
                "size": 0x10,
                "name": "sub_00020500",
                "section": ".text",
                "num_instructions": 1,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            xbe_path = os.path.join(directory, "test.xbe")
            functions_path = os.path.join(directory, "functions.json")
            baseline_path = os.path.join(directory, "baseline.json")
            delta_path = os.path.join(directory, "delta.json")
            with open(xbe_path, "wb") as output:
                output.write(xbe)
            with open(functions_path, "w") as output:
                json.dump(functions, output)
            with open(baseline_path, "w") as output:
                json.dump([{
                    "start": f"0x{TRUNCATED_START:08X}",
                    "end": f"0x{FULL_END:08X}",
                }], output)
            with open(delta_path, "w") as output:
                json.dump([{
                    "start": f"0x{UNRELATED:08X}",
                    "end": f"0x{DISJOINT_END:08X}",
                }], output)

            translator = BatchTranslator(
                xbe_path=xbe_path,
                func_json_path=functions_path,
                recovery_json_path=[baseline_path, delta_path],
            )

        self.assertEqual(
            translator.func_db[TRUNCATED_START]["end"], FULL_END)
        self.assertIn(UNRELATED, translator.func_db)
        self.assertEqual(
            translator.translator.recovered_function_starts, {UNRELATED})
        self.assertEqual(
            translator.translator.extended_function_starts,
            {TRUNCATED_START})

    def test_explicit_coalescence_removes_false_interior_start(self):
        translator, func_db = make_translator({
            TRUNCATED_START + 9: {
                "_addr": TRUNCATED_START + 9,
                "start": f"0x{TRUNCATED_START + 9:08X}",
                "end": FULL_END,
                "size": FULL_END - TRUNCATED_START - 9,
                "name": "sub_00020009",
                "section": ".text",
                "num_instructions": 2,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
        })

        translator.adopt_external_functions([{
            "start": TRUNCATED_START,
            "end": FULL_END,
            "coalesce_starts": [TRUNCATED_START + 9],
        }])

        self.assertNotIn(TRUNCATED_START + 9, func_db)
        self.assertEqual(func_db[TRUNCATED_START]["end"], FULL_END)
        self.assertEqual(
            func_db[TRUNCATED_START]["detection_method"],
            "external_coalescence")
        self.assertEqual(
            translator.coalesced_function_starts, {TRUNCATED_START})
        self.assertIn(TRUNCATED_START, translator._recovered_cfg)

    def test_last_function_coalescence_uses_authenticated_end(self):
        """A final detected body has no later seed to use as a decode bound."""
        false_start = TRUNCATED_START + 9
        translator, func_db = make_translator({
            false_start: _thunk(false_start, FULL_END),
        })
        del func_db[NEXT]
        del func_db[SENTINEL]

        translator.adopt_external_functions([{
            "start": TRUNCATED_START,
            "end": FULL_END,
            "coalesce_starts": [false_start],
        }])

        self.assertNotIn(false_start, func_db)
        self.assertEqual(func_db[TRUNCATED_START]["end"], FULL_END)
        self.assertIn(TRUNCATED_START, translator._recovered_cfg)

    def test_coalescence_reads_jump_table_at_owned_end(self):
        translator, func_db = make_translator({
            TABLE_BODY_START: {
                "_addr": TABLE_BODY_START,
                "start": f"0x{TABLE_BODY_START:08X}",
                "end": TABLE_CASE,
                "size": TABLE_CASE - TABLE_BODY_START,
                "name": "sub_00020020",
                "section": ".text",
                "num_instructions": 2,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
            TABLE_CASE: {
                "_addr": TABLE_CASE,
                "start": f"0x{TABLE_CASE:08X}",
                "end": TABLE_BODY_END,
                "size": TABLE_BODY_END - TABLE_CASE,
                "name": "sub_00020029",
                "section": ".text",
                "num_instructions": 4,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
        })

        translator.adopt_external_functions([{
            "start": TABLE_BODY_START,
            "end": TABLE_BODY_END,
            "coalesce_starts": [TABLE_CASE],
        }])

        self.assertNotIn(TABLE_CASE, func_db)
        recovered = translator._recovered_cfg[TABLE_BODY_START]
        self.assertEqual(recovered["end"], TABLE_BODY_END)
        self.assertEqual(
            recovered["jump_tables"][TABLE_ADDRESS],
            [TABLE_CASE, TABLE_CASE + 2])

    def test_coalescence_uses_exact_external_bridge_for_padding(self):
        translator, func_db = make_translator({
            BRIDGE_BODY_START: {
                "_addr": BRIDGE_BODY_START,
                "start": f"0x{BRIDGE_BODY_START:08X}",
                "end": BRIDGE_GAP,
                "size": BRIDGE_GAP - BRIDGE_BODY_START,
                "name": "sub_00020040",
                "section": ".text",
                "num_instructions": 1,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
            BRIDGE_CASE: {
                "_addr": BRIDGE_CASE,
                "start": f"0x{BRIDGE_CASE:08X}",
                "end": BRIDGE_BODY_END,
                "size": BRIDGE_BODY_END - BRIDGE_CASE,
                "name": "sub_00020044",
                "section": ".text",
                "num_instructions": 1,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
        })

        translator.adopt_external_functions([{
            "start": BRIDGE_BODY_START,
            "end": BRIDGE_BODY_END,
            "coalesce_starts": [BRIDGE_CASE],
            "coalesce_bridges": [BRIDGE_GAP],
        }])

        self.assertNotIn(BRIDGE_CASE, func_db)
        self.assertEqual(
            [insn.address for insn in
             translator._recovered_cfg[BRIDGE_BODY_START]["instructions"]],
            [BRIDGE_BODY_START, BRIDGE_GAP, BRIDGE_CASE])

    def test_coalescence_accepts_lea_alignment_padding(self):
        """The retail XBE pads with lea reg, [reg] rather than nop."""
        translator, func_db = make_translator({
            LEA_BODY_START: _thunk(LEA_BODY_START, LEA_GAP),
            LEA_CASE: _thunk(LEA_CASE, LEA_BODY_END),
        })

        translator.adopt_external_functions([{
            "start": LEA_BODY_START,
            "end": LEA_BODY_END,
            "coalesce_starts": [LEA_CASE],
            "coalesce_bridges": [LEA_GAP],
        }])

        self.assertNotIn(LEA_CASE, func_db)
        self.assertEqual(
            [insn.address for insn in
             translator._recovered_cfg[LEA_BODY_START]["instructions"]],
            [LEA_BODY_START, LEA_GAP, LEA_CASE])

    def test_coalescence_rejects_bridge_over_live_instruction(self):
        """A bridge may only name alignment padding, not real code."""
        translator, _ = make_translator({
            LIVE_BODY_START: _thunk(LIVE_BODY_START, LIVE_GAP),
            LIVE_CASE: _thunk(LIVE_CASE, LIVE_BODY_END),
        })

        with self.assertRaisesRegex(ValueError, "is not a multi-byte no-op"):
            translator.adopt_external_functions([{
                "start": LIVE_BODY_START,
                "end": LIVE_BODY_END,
                "coalesce_starts": [LIVE_CASE],
                "coalesce_bridges": [LIVE_GAP],
            }])

    def test_coalescence_rejects_bridge_padding_to_unnamed_start(self):
        """Padding must land exactly on a start the entry names."""
        translator, _ = make_translator({
            UNNAMED_BODY_START: _thunk(UNNAMED_BODY_START, UNNAMED_GAP),
            UNNAMED_CASE: _thunk(UNNAMED_CASE, UNNAMED_BODY_END),
        })

        with self.assertRaisesRegex(ValueError, "not a coalesced start"):
            translator.adopt_external_functions([{
                "start": UNNAMED_BODY_START,
                "end": UNNAMED_BODY_END,
                "coalesce_starts": [UNNAMED_CASE],
                "coalesce_bridges": [UNNAMED_GAP],
            }])

    def test_coalescence_closes_alignment_padding_without_a_bridge(self):
        """Padding mid-body coalesces with nothing declared for it.

        This is the retail shape: the pad aligns an interior branch target, so
        it lands on ordinary code rather than on a named false start and no
        caller can name it as a bridge.
        """
        translator, func_db = make_translator({
            AUTOPAD_BODY_START: _thunk(AUTOPAD_BODY_START, AUTOPAD_GAP),
            AUTOPAD_CASE: _thunk(AUTOPAD_CASE, AUTOPAD_BODY_END),
        })

        translator.adopt_external_functions([{
            "start": AUTOPAD_BODY_START,
            "end": AUTOPAD_BODY_END,
            "coalesce_starts": [AUTOPAD_CASE],
        }])

        self.assertNotIn(AUTOPAD_CASE, func_db)
        self.assertEqual(
            [insn.address for insn in
             translator._recovered_cfg[AUTOPAD_BODY_START]["instructions"]],
            [AUTOPAD_BODY_START, AUTOPAD_GAP, AUTOPAD_RESUME, AUTOPAD_CASE])

    def test_coalescence_still_rejects_a_gap_of_real_code(self):
        """Automatic closure may only span padding, never unreached code."""
        translator, _ = make_translator({
            AUTOLIVE_BODY_START: _thunk(AUTOLIVE_BODY_START, AUTOLIVE_GAP),
            AUTOLIVE_GAP: _thunk(AUTOLIVE_GAP, AUTOLIVE_BODY_END),
        })

        with self.assertRaisesRegex(ValueError, "CFG gap at"):
            translator.adopt_external_functions([{
                "start": AUTOLIVE_BODY_START,
                "end": AUTOLIVE_BODY_END,
                "coalesce_starts": [AUTOLIVE_GAP],
            }])

    def test_coalescence_updates_later_disjoint_bounds(self):
        interior = TRUNCATED_START + 9
        translator, func_db = make_translator({
            interior: {
                "_addr": interior,
                "start": f"0x{interior:08X}",
                "end": FULL_END,
                "size": FULL_END - interior,
                "name": "sub_00020009",
                "section": ".text",
                "num_instructions": 2,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
        })
        translator._recovered_cfg[interior] = {"end": FULL_END}

        translator.adopt_external_functions([
            {
                "start": TRUNCATED_START,
                "end": FULL_END,
                "coalesce_starts": [interior],
            },
            {"start": interior + 1, "end": FULL_END},
        ])

        self.assertNotIn(interior, translator._recovered_cfg)
        self.assertNotIn(interior + 1, func_db)

    def test_coalescence_requires_exact_interior_start_census(self):
        translator, _ = make_translator({
            TRUNCATED_START + 9: {
                "_addr": TRUNCATED_START + 9,
                "start": f"0x{TRUNCATED_START + 9:08X}",
                "end": FULL_END,
                "size": FULL_END - TRUNCATED_START - 9,
                "name": "sub_00020009",
                "section": ".text",
                "num_instructions": 2,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
        })

        with self.assertRaisesRegex(
                ValueError, "current interior starts"):
            translator.adopt_external_functions([{
                "start": TRUNCATED_START,
                "end": FULL_END,
                "coalesce_starts": [TRUNCATED_START + 8],
            }])

    def test_same_start_exact_extension_is_adopted(self):
        """The old code silently skipped any entry whose start existed."""
        translator, func_db = make_translator()

        translator.adopt_external_functions(
            [{"start": TRUNCATED_START, "end": FULL_END}])

        entry = func_db[TRUNCATED_START]
        self.assertEqual(entry["end"], FULL_END)
        self.assertEqual(entry["size"], FULL_END - TRUNCATED_START)
        self.assertEqual(entry["detection_method"], "external_extension")
        self.assertEqual(entry["num_instructions"], 6)
        self.assertIn(TRUNCATED_START, translator.extended_function_starts)
        recovered = translator._recovered_cfg[TRUNCATED_START]
        self.assertEqual(recovered["end"], FULL_END)
        # No new function was invented for the extended tail.
        self.assertEqual(set(func_db), {TRUNCATED_START, NEXT, SENTINEL})
        self.assertEqual(translator.recovered_function_starts, set())

    def test_extension_closes_alignment_padding_sequence(self):
        translator, func_db = make_translator({
            EXT_AUTOPAD_BODY_START: _thunk(
                EXT_AUTOPAD_BODY_START, EXT_AUTOPAD_GAP),
        })

        translator.adopt_external_functions([{
            "start": EXT_AUTOPAD_BODY_START,
            "end": EXT_AUTOPAD_BODY_END,
        }])

        self.assertEqual(
            func_db[EXT_AUTOPAD_BODY_START]["end"], EXT_AUTOPAD_BODY_END)
        self.assertEqual(
            [insn.address for insn in
             translator._recovered_cfg[
                 EXT_AUTOPAD_BODY_START]["instructions"]],
            [EXT_AUTOPAD_BODY_START, EXT_AUTOPAD_GAP,
             EXT_AUTOPAD_GAP + 7, EXT_AUTOPAD_RESUME])

    def test_equal_end_is_inert(self):
        translator, func_db = make_translator()
        before = dict(func_db[TRUNCATED_START])

        translator.adopt_external_functions(
            [{"start": TRUNCATED_START, "end": DETECTED_END}])

        self.assertEqual(func_db[TRUNCATED_START], before)
        self.assertEqual(translator.extended_function_starts, set())
        self.assertEqual(translator._recovered_cfg, {})

    def test_shrink_rejects(self):
        translator, func_db = make_translator()

        with self.assertRaisesRegex(ValueError, "shrinks detected end"):
            translator.adopt_external_functions(
                [{"start": TRUNCATED_START, "end": DETECTED_END - 2}])
        self.assertEqual(func_db[TRUNCATED_START]["end"], DETECTED_END)

    def test_cross_next_function_rejects(self):
        translator, _ = make_translator()

        with self.assertRaisesRegex(
                ValueError, "crosses next function start"):
            translator.adopt_external_functions(
                [{"start": TRUNCATED_START, "end": NEXT + 4}])

    def test_interior_function_rejects(self):
        """A start inside the requested extent bounds the extension."""
        interior = TRUNCATED_START + 0x4
        translator, _ = make_translator({
            interior: {
                "_addr": interior,
                "start": f"0x{interior:08X}",
                "end": interior + 2,
                "size": 2,
                "name": "sub_00020004",
                "section": ".text",
                "num_instructions": 1,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
        })

        with self.assertRaisesRegex(
                ValueError, "crosses next function start"):
            translator.adopt_external_functions(
                [{"start": TRUNCATED_START, "end": FULL_END}])

    def test_preceding_function_overlapping_extent_rejects(self):
        """An earlier function already owning the new bytes must reject."""
        earlier = TRUNCATED_START - 0x10
        translator, _ = make_translator({
            earlier: {
                "_addr": earlier,
                "start": f"0x{earlier:08X}",
                "end": FULL_END - 1,
                "size": FULL_END - 1 - earlier,
                "name": "sub_0001FFF0",
                "section": ".text",
                "num_instructions": 1,
                "detection_method": "seed_vtable_thunk",
                "has_prologue": False,
                "calls_to": [],
                "called_by": [],
            },
        })

        with self.assertRaisesRegex(
                ValueError, "overlaps detected function"):
            translator.adopt_external_functions(
                [{"start": TRUNCATED_START, "end": FULL_END}])

    def test_unproven_end_rejects_when_cfg_stops_short(self):
        """Requesting past the decodable body must not be adopted."""
        translator, func_db = make_translator()

        with self.assertRaises(ValueError):
            translator.adopt_external_functions(
                [{"start": TRUNCATED_START, "end": FULL_END + 0x10}])
        self.assertEqual(func_db[TRUNCATED_START]["end"], DETECTED_END)
        self.assertEqual(translator.extended_function_starts, set())

    def test_overrun_end_rejects_when_end_splits_an_instruction(self):
        translator, _ = make_translator()

        with self.assertRaisesRegex(
                ValueError, "past the requested end"):
            translator.adopt_external_functions(
                [{"start": TRUNCATED_START, "end": FULL_END - 1}])

    def test_disjoint_entry_is_adopted_alongside_an_extension(self):
        """Disjoint recovery keeps adding new bodies, unchanged."""
        translator, func_db = make_translator()
        before_next = dict(func_db[NEXT])

        translator.adopt_external_functions([
            {"start": TRUNCATED_START, "end": FULL_END},
            {"start": UNRELATED, "end": DISJOINT_END},
        ])

        # The disjoint entry became a new function.
        self.assertIn(UNRELATED, func_db)
        self.assertIn(UNRELATED, translator.recovered_function_starts)
        self.assertEqual(func_db[UNRELATED]["end"], DISJOINT_END)
        self.assertEqual(
            func_db[UNRELATED]["detection_method"], "external_boundary")
        # It is an addition, not an extension.
        self.assertNotIn(UNRELATED, translator.extended_function_starts)
        self.assertEqual(
            translator.extended_function_starts, {TRUNCATED_START})
        # The unrelated detected function is untouched either way.
        self.assertEqual(func_db[NEXT], before_next)
        self.assertNotIn(NEXT, translator.extended_function_starts)
        self.assertNotIn(NEXT, translator._recovered_cfg)


if __name__ == "__main__":
    unittest.main()
