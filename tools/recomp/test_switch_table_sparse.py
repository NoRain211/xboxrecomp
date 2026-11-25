"""
Self-check for switch tables with zero holes before later arms.

Run: py -3 tools/recomp/test_switch_table_sparse.py

MSVC splits one dispatch across two adjacent tables when several indices
share a default arm. The second table then has null slots where the first
table already routed those indices:

    jmp [eax*4 + 0x2B37B8]   ; four indices land on 0x2B3230
    ...
    0x2B3230: sub ecx, 2
              jmp [ecx*4 + 0x2B37DC]   ; slots 2 and 4 are null

Stopping at the first null slot dropped the live arms after it, so the
indexed jump remained an indirect tail call and DOA2U stopped when a player
changed the online lobby's life or timer setting.

The emitted switch compares the loaded word against its arms, so a hole can
never be selected. The scan only needs to keep going to the next arm.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.recomp import config  # noqa: E402
from tools.recomp.translator import FunctionTranslator  # noqa: E402

BASE = 0x00010000
TABLE = BASE + 16
ARMS = [BASE + 48, BASE + 54, BASE + 60, BASE + 66]
ENTRIES = [ARMS[0], 0, ARMS[1], ARMS[2], 0, ARMS[3]]


def _translate(image):
    config._install(
        [config.Section(".text", BASE, len(image), 0x0000, len(image), True)],
        entry_point=BASE, kernel_thunk_addr=BASE, origin="sparse-switch-test")
    db = {BASE: {"start": f"0x{BASE:08X}", "end": BASE + len(image),
                 "_addr": BASE, "size": len(image)}}
    return FunctionTranslator(image, db).translate_function(BASE, db[BASE])


def _image(entries):
    """jmp [eax*4 + TABLE]; pad to TABLE; entries; pad to arms; arms."""
    code = b"\xFF\x24\x85" + TABLE.to_bytes(4, "little")
    code += b"\x90" * (TABLE - BASE - len(code))
    code += b"".join(entry.to_bytes(4, "little") for entry in entries)
    for number, arm in enumerate(ARMS, start=1):
        code += b"\x90" * (arm - BASE - len(code))
        code += b"\xB8" + number.to_bytes(4, "little") + b"\xC3"
    return code


def test_zero_holes_are_skipped():
    code = _translate(_image(ENTRIES))
    assert f"switch: {len(ARMS)} entries" in code, code
    for arm in ARMS:
        assert f"goto loc_{arm:08X};" in code, f"arm {arm:#x} missing\n{code}"


def test_non_code_word_still_ends_the_table():
    # A zero hole must not make an unrelated data word start scanning again.
    entries = [ARMS[0], 0, 0x01010000, ARMS[1]]
    code = _translate(_image(entries))
    assert "switch: 1 entries" in code, code
    assert f"goto loc_{ARMS[0]:08X};" in code, code
    assert f"goto loc_{ARMS[1]:08X};" not in code, code


if __name__ == "__main__":
    test_zero_holes_are_skipped()
    test_non_code_word_still_ends_the_table()
    print("ok")
