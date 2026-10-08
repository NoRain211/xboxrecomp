"""Comparison snapshots must survive a join of different CMP operands."""
from tools.recomp import config
from tools.recomp.translator import FunctionTranslator, _merge_flag_states
from tools.recomp.lifter import JOINED
from tools.recomp.disasm import Operand

BASE = 0x10000

def translate_join(consumer=bytes.fromhex('0f95c0c3')):
    # test ecx,ecx; jz alternate; cmp eax,edx; jmp join; nop;
    # alternate: cmp ebx,esi; join: consumer.
    image = bytes.fromhex('85c9740539d0eb039039f3') + consumer
    config._install([config.Section('.text', BASE, len(image), 0, len(image), True)],
                              entry_point=BASE, kernel_thunk_addr=BASE,
                              origin='flag-join-test')
    db = {BASE: {'start': hex(BASE), 'end': BASE + len(image),
                 '_addr': BASE, 'size': len(image)}}
    return FunctionTranslator(image, db).translate_function(BASE, db[BASE])

def test_different_cmp_operands_join_for_setne():
    code = translate_join()
    assert 'CMP_NE(_fa, _fb)' in code, code
    assert '_flags /* setne */' not in code
    # One condition fits both paths, so no setter tag is needed.
    assert '_fk' not in code, code

def test_different_cmp_operands_join_for_cmovne():
    code = translate_join(bytes.fromhex('0f45c7c3'))
    assert 'if (CMP_NE(_fa, _fb)) eax = edi;' in code, code

def test_unknown_path_is_not_guessed():
    assert _merge_flag_states([None, ('cmp', [])]) is None

def test_mixed_operations_join_as_both_states():
    a = Operand(type='reg', reg='eax')
    b = Operand(type='reg', reg='edx')
    merged = _merge_flag_states([('cmp', [a, b]), ('test', [a, b])])
    assert merged == (JOINED, [('cmp', [a, b]), ('test', [a, b])])

def translate_width_join(consumer):
    # test ecx,ecx; jz narrow; cmp ecx,eax; jmp join;
    # narrow: cmp al,1; join: consumer +1; inc eax; ret.
    image = bytes.fromhex('85c9740439c1eb023c01') + consumer + bytes.fromhex('40c3')
    config._install([config.Section('.text', BASE, len(image), 0, len(image), True)],
                    entry_point=BASE, kernel_thunk_addr=BASE,
                    origin='flag-join-test')
    db = {BASE: {'start': hex(BASE), 'end': BASE + len(image),
                 '_addr': BASE, 'size': len(image)}}
    return FunctionTranslator(image, db).translate_function(BASE, db[BASE])

def test_mixed_width_compares_join_for_jne():
    code = translate_width_join(bytes.fromhex('7501'))
    assert 'CMP_NE(_fa, _fb)' in code, code
    assert '_flags /* jne' not in code, code

def test_mixed_widths_pick_the_sign_bit_by_setter_tag():
    # SF is the top bit at each compare's own width; no one expression fits,
    # so each compare leaves its tag in _fk and js picks its own sign bit.
    code = translate_width_join(bytes.fromhex('7801'))
    assert '_flags /* js' not in code, code
    assert 'uint32_t _fk = 0;' in code, code
    assert code.count('/* flag setter tag */') >= 3, code
    assert '(_fk == 0x' in code, code

def translate_or_join(consumer):
    # test ecx,ecx; jz other; mov ecx,edx; or ecx,ebx; jmp join;
    # other: or edx,ebx; join: consumer +1; inc eax; ret.
    image = bytes.fromhex('85c9740689d109d9eb0209da') + consumer + bytes.fromhex('40c3')
    config._install([config.Section('.text', BASE, len(image), 0, len(image), True)],
                    entry_point=BASE, kernel_thunk_addr=BASE,
                    origin='flag-join-test')
    db = {BASE: {'start': hex(BASE), 'end': BASE + len(image),
                 '_addr': BASE, 'size': len(image)}}
    return FunctionTranslator(image, db).translate_function(BASE, db[BASE])

def test_ors_into_different_registers_join_for_je():
    # DOA3's title input logic ORs a pad mask into ecx on one path and edx
    # on the other, then branches on ZF. Both results land in _fa, so the
    # branch must test it; the never-taken fallback made the title act on
    # a START that was not pressed.
    for consumer, cond in (('7401', '(_fa == 0)'), ('7501', '(_fa != 0)')):
        code = translate_or_join(bytes.fromhex(consumer))
        assert f'if ({cond}) goto' in code, code
        assert '_flags /* j' not in code, code
