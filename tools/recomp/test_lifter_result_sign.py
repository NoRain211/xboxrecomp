import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from .test_lifter_test_sign import _lift


class ResultSignTest(unittest.TestCase):
    def test_emitted_sign_conditions(self):
        compiler = shutil.which('gcc') or shutil.which('clang')
        self.assertIsNotNone(compiler, 'A C compiler is required')
        functions, checks = [], []
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
            static uint32_t _fa, _fb; static int32_t _fas, _fbs;
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
