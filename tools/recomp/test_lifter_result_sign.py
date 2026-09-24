import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from .disasm import Operand
from .lifter import _make_condition
from .test_lifter_test_sign import _lift


class ResultSignTest(unittest.TestCase):
    def test_emitted_sign_conditions(self):
        compiler = shutil.which('gcc') or shutil.which('clang')
        self.assertIsNotNone(compiler, 'A C compiler is required')
        setters = ('add', 'sub', 'adc', 'sbb', 'and', 'or', 'xor',
                   'inc', 'dec', 'neg', 'shl', 'shr', 'sar', 'shld', 'shrd')
        functions, checks = [], []
        for bits, register in ((8, 'al'), (8, 'ah'), (16, 'ax'), (32, 'eax')):
            for setter in setters:
                operands = [Operand(type='reg', reg=register), Operand(type='imm', imm=1)]
                if setter in ('inc', 'dec', 'neg'):
                    operands = operands[:1]
                for branch in ('js', 'jns'):
                    condition = _make_condition(branch, setter, operands)[0]
                    name = f'{setter}_{register}_{branch}'
                    value = 'v << 8' if register == 'ah' else 'v'
                    functions.append(f'static unsigned {name}(uint32_t v) {{ eax={value}; return {condition}; }}')
                    sign = 1 << (bits - 1)
                    for value in (0, sign-1, sign, (1 << bits)-1):
                        expected = int(bool(value & sign) == (branch == 'js'))
                        checks.append(f'CHECK({name}({value}u), {expected}u);')
        # Synthetic decrement, flag-preserving store, and sign branch.
        raw = bytes.fromhex('fec88887180000007800')
        lifted = _lift(raw)
        functions.append(f'''static unsigned decrement_store(uint32_t v) {{
            eax=v; edi=0; {lifted}
            return 0; loc_{len(raw):08X}: return 1;
        }}''')
        checks.extend(('CHECK(decrement_store(0),1);', 'CHECK(decrement_store(1),0);',
                       'CHECK(decrement_store(0x80),0);', 'CHECK(decrement_store(0x81),1);'))
        source = r'''
            #include <stdint.h>
            #include <stdio.h>
            static uint32_t eax, edi;
            static uint8_t memory[256];
            #define LO8(x) ((uint8_t)(x))
            #define HI8(x) ((uint8_t)((x)>>8))
            #define LO16(x) ((uint16_t)(x))
            #define SET_LO8(x,v) ((x)=((x)&0xffffff00u)|(uint8_t)(v))
            #define MEM8(a) memory[a]
            #define CHECK(actual,expected) do { if ((actual)!=(expected)) { if (errors<4) fprintf(stderr,"%s failed\n",#actual); ++errors; } } while(0)
        ''' + '\n'.join(functions)
        source += '\nint main(void) { unsigned errors=0;\n'+'\n'.join(checks)+'\nreturn errors!=0; }'
        with tempfile.TemporaryDirectory(prefix='result-sign-') as temp:
            path = Path(temp)/'check.c'
            exe = Path(temp)/'check.exe'
            path.write_text(source)
            subprocess.run([compiler, '-std=c11', str(path), '-o', str(exe)],
                           check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
