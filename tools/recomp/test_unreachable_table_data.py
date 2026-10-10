"""An inline table must not create flag consumers or phantom callers."""
from tools.recomp import config
from tools.recomp.translator import FunctionTranslator


def test_table_after_tail_is_not_lifted_but_switch_arms_are():
    base, table = 0x10000, 0x10020
    # jmp [eax*4+table]; arm 0: ret; arm 1: ret; dead jo; padding; table
    body = (bytes.fromhex("ff2485") + table.to_bytes(4, "little")
            + bytes.fromhex("c3c370fe")).ljust(32, b"\x90")
    body += (base + 7).to_bytes(4, "little") + (base + 8).to_bytes(4, "little")
    config._install([config.Section('.text', base, len(body), 0, len(body), True)],
                    entry_point=base, kernel_thunk_addr=base, origin='table-test')
    info = {'start': hex(base), 'end': base + len(body), '_addr': base}
    translator = FunctionTranslator(body, {base: info})
    code = translator.translate_function(base, info)
    assert 'loc_00010007: ;' in code and 'loc_00010008: ;' in code
    assert '_flags /* jo' not in code
    instructions, _ = translator.decode_function(base, base + len(body))
    assert [i.address for i in instructions] == [base, base + 7, base + 8]
