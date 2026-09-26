"""A switch arm outside its dispatching function needs a callable entry."""
import unittest
from unittest.mock import patch

from . import config
from .translator import FunctionTranslator

TEXT = 0x00020000
SPLIT = TEXT + 0x10   # a false function start the switch runs across
ARM = TEXT + 0x30     # an arm inside that piece, with no entry of its own
TABLE = TEXT + 0x40


def entry(start, end):
    return {"_addr": start, "start": f"0x{start:08X}", "end": end,
            "section": ".text"}


class JumpTableArmEntryTest(unittest.TestCase):
    def setUp(self):
        sections = patch.object(config, "_SECTIONS", [
            config.Section(".text", TEXT, 0x100, 0, 0x100, True),
        ])
        sections.start()
        self.addCleanup(sections.stop)

    def translator(self, first_end):
        image = bytearray(b"\xCC" * 0x100)
        # jmp dword ptr [eax*4 + TABLE]
        image[0:7] = bytes.fromhex("ff2485") + TABLE.to_bytes(4, "little")
        image[SPLIT - TEXT] = 0xC3
        image[ARM - TEXT] = 0xC3
        for index, target in enumerate((SPLIT, ARM)):
            offset = TABLE - TEXT + index * 4
            image[offset:offset + 4] = target.to_bytes(4, "little")
        return FunctionTranslator(bytes(image), {
            TEXT: entry(TEXT, first_end),
            SPLIT: entry(SPLIT, TABLE),
        })

    def test_arm_inside_a_split_piece_gets_an_entry(self):
        translator = self.translator(SPLIT)
        self.assertEqual(translator.discover_jump_table_entries(), {ARM})
        self.assertEqual(translator.func_db[ARM]["end"], TABLE)
        self.assertEqual(translator.func_db[ARM]["called_by"], [TEXT])
        self.assertEqual(translator.func_db[SPLIT]["end"], TABLE)

    def test_switch_inside_its_function_adds_nothing(self):
        translator = self.translator(TABLE)
        self.assertEqual(translator.discover_jump_table_entries(), set())


if __name__ == "__main__":
    unittest.main()
