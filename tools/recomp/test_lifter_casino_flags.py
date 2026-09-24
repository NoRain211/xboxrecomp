"""Execute lifted comparison flags across MOV and fallthrough boundaries."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from .disasm import Disassembler
from .lifter import Lifter, carry_crosses_blocks, lift_basic_block
from .test_lifter_test_sign import _lift


class CasinoFlagsTest(unittest.TestCase):
    def execute(self, source):
        compiler = shutil.which("gcc") or shutil.which("clang")
        self.assertIsNotNone(compiler, "C compiler required")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.c"
            exe = path.with_suffix(".exe")
            path.write_text("""
                #include <stdint.h>
                #include <assert.h>
                #define LO8(x) ((uint8_t)(x))
                #define LO16(x) ((uint16_t)(x))
                #define SET_LO8(x,v) ((x)=((x)&0xffffff00u)|(uint8_t)(v))
                #define SET_LO16(x,v) ((x)=((x)&0xffff0000u)|(uint16_t)(v))
                #define TEST_S(a,b) ((int32_t)((a)&(b))<0)
                #define CMP_G(a,b) ((int32_t)(a)>(int32_t)(b))
                #define CMP_NE(a,b) ((a)!=(b))
            """ + source)
            subprocess.run([compiler, "-std=c11", str(path), "-o", str(exe)],
                           check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_compare_carry_survives_mov_and_consumes_original_operands(self):
        functions = []
        checks = []
        # CMP/TEST at each width, with a flag-preserving MOV overwriting RHS.
        for index, (bits, cmp_op, test_op) in enumerate((
                (8, "38c8", "84c8"), (16, "6639c8", "6685c8"),
                (32, "39c8", "85c8"))):
            mask = (1 << bits) - 1
            for name, opcode in (("cmp", cmp_op), ("test", test_op)):
                fn = f"check_{name}_{index}"
                lifted = _lift(bytes.fromhex(opcode + "b90000000018d2fec2"))
                functions.append(f"""
                    static uint32_t {fn}(uint32_t eax, uint32_t ecx) {{
                        uint32_t edx=0; int _cf=1;
                        {lifted}
                        return LO8(edx);
                    }}
                """)
                expected = f"((values[i]&{mask}u)>=(values[j]&{mask}u))" if name == "cmp" else "1"
                checks.append(f"""
                    for(unsigned i=0;i<6;i++) for(unsigned j=0;j<6;j++)
                        assert({fn}(values[i],values[j]) == {expected});
                """)
        self.execute("\n".join(functions) + """
            int main(void) {
                uint32_t values[]={0,1,255,65535,0x80000000u,0xffffffffu};
        """ + "\n".join(checks) + "}")

    def test_test_flags_survive_fallthrough_register_replacement(self):
        # TEST EAX,EAX; JS done; MOV EAX,99999999; JG done; MOV EAX,0; done:
        raw = bytes.fromhex("85c0780eb8ffe0f5057f07b800000000eb0090")
        disasm = Disassembler()
        insns = disasm.disassemble_function(raw, 0, len(raw))
        lifter = Lifter()
        lifter.func_end = len(raw) + 1
        lines, state = [], None
        for block in disasm.build_basic_blocks(insns, 0, len(raw)):
            stmts, state = lift_basic_block(lifter, block, state)
            lines += [f"loc_{block.start:08X}: ;"] + stmts
        self.execute("uint32_t check(uint32_t eax) {\n" + "\n".join(lines) + """
                return eax;
            }
            int main(void) {
                assert(check(0)==0);
                assert(check(1)==99999999);
                assert(check(0xffffffffu)==0xffffffffu);
            }
        """)

    def lift_function(self, raw):
        disasm = Disassembler()
        insns = disasm.disassemble_function(raw, 0, len(raw))
        blocks = disasm.build_basic_blocks(insns, 0, len(raw))
        lifter = Lifter()
        lifter.func_end = len(raw) + 1
        lifter.cross_block_carry = carry_crosses_blocks(lifter, blocks)
        lines, state = [], None
        for block in blocks:
            stmts, state = lift_basic_block(lifter, block, state)
            lines += [f"loc_{block.start:08X}: ;"] + stmts
        return "\n".join(lines)

    def test_compare_carry_reaches_sbb_in_branch_target(self):
        # CMP DL,BL; JNE L; XOR EAX,EAX; JMP done; L: SBB EAX,EAX;
        # SBB EAX,-1; done: NOP -- the tail of an MSVC string compare.
        lifted = self.lift_function(
            bytes.fromhex("38da750431c0eb0519c083d8ff90"))
        self.execute("""
            static uint32_t check(uint32_t edx, uint32_t ebx) {
                uint32_t eax=0; int _cf=0;
        """ + lifted + """
                return eax;
            }
            int main(void) {
                uint32_t values[]={0,1,0x41,0x7f,0x80,0xff,0x1ff};
                for(unsigned i=0;i<7;i++) for(unsigned j=0;j<7;j++) {
                    uint8_t a=(uint8_t)values[i], b=(uint8_t)values[j];
                    assert(check(values[i],values[j]) ==
                           (a<b ? 0xffffffffu : a>b ? 1u : 0u));
                }
            }
        """)

    def test_xor_clears_carry_for_sbb(self):
        # XOR EAX,ECX; SBB EAX,EAX with a stale carry of 1.
        lifted = _lift(bytes.fromhex("31c819c0"))
        self.execute("""
            static uint32_t check(uint32_t eax, uint32_t ecx) {
                int _cf=1;
        """ + lifted + """
                return eax;
            }
            int main(void) { assert(check(5,3)==0); }
        """)


if __name__ == "__main__":
    unittest.main()
