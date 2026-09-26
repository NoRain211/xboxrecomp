"""A switch arm outside its dispatching function needs a callable entry."""
import unittest
from unittest.mock import patch

from . import config
from .translator import FunctionTranslator

TEXT = 0x00020000
SPLIT = TEXT + 0x10    # a false function start the switch runs across
PIECE = TEXT + 0x34    # another false start, inside the arm's code
ARM = TEXT + 0x30      # an arm with no entry: jmp to PIECE + 4, then ret
TABLE = TEXT + 0x40    # the split switch: SPLIT, ARM
NEXT_TABLE = TEXT + 0x48  # SPLIT's own switch, which follows it


def entry(start, end):
    return {"_addr": start, "start": f"0x{start:08X}", "end": end,
            "section": ".text"}


def put(image, va, data):
    image[va - TEXT:va - TEXT + len(data)] = data


class JumpTableArmEntryTest(unittest.TestCase):
    def setUp(self):
        sections = patch.object(config, "_SECTIONS", [
            config.Section(".text", TEXT, 0x100, 0, 0x100, True),
        ])
        sections.start()
        self.addCleanup(sections.stop)

    def translator(self, func_db):
        image = bytearray(b"\xCC" * 0x100)
        put(image, TEXT, bytes.fromhex("ff2485") + TABLE.to_bytes(4, "little"))
        put(image, SPLIT,
            bytes.fromhex("ff2485") + NEXT_TABLE.to_bytes(4, "little"))
        put(image, SPLIT + 0x10, b"\xC3")
        put(image, SPLIT + 0x14, b"\xC3")
        put(image, ARM, bytes.fromhex("eb06"))  # jmp PIECE + 4
        put(image, PIECE + 4, b"\xC3")
        for table, targets in ((TABLE, (SPLIT, ARM)),
                               (NEXT_TABLE, (SPLIT + 0x10, SPLIT + 0x14))):
            put(image, table, b"".join(t.to_bytes(4, "little") for t in targets))
        return FunctionTranslator(bytes(image), func_db)

    def test_arm_gets_an_entry_covering_its_reachable_code(self):
        translator = self.translator({
            TEXT: entry(TEXT, SPLIT),
            SPLIT: entry(SPLIT, PIECE),
            PIECE: entry(PIECE, TABLE),
        })
        # The adjacent table's arms belong to SPLIT's own switch.
        self.assertEqual(translator.discover_jump_table_entries(), {ARM})
        self.assertEqual(translator.func_db[ARM]["end"], PIECE + 5)
        self.assertEqual(translator.func_db[ARM]["called_by"], [TEXT])
        self.assertEqual(translator.func_db[PIECE]["end"], TABLE)

    def test_switch_inside_its_function_adds_nothing(self):
        translator = self.translator({TEXT: entry(TEXT, TABLE)})
        self.assertEqual(translator.discover_jump_table_entries(), set())


if __name__ == "__main__":
    unittest.main()