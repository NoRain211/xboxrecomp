import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from .disasm import BasicBlock, Disassembler, Operand
from .lifter import Lifter, _make_condition, lift_basic_block


def _lift(raw):
    instructions = Disassembler().disassemble_function(raw, 0, len(raw))
    lifter = Lifter()
    lifter.func_end = len(raw) + 1
    return "\n".join(lift_basic_block(
        lifter, BasicBlock(start=0, instructions=instructions))[0])


class TestSignWidthTest(unittest.TestCase):
    def test_dword_output_is_unchanged(self):
        operands = [Operand(type="reg", reg="ecx")] * 2
        self.assertEqual(_make_condition("js", "test", operands)[0],
                         "TEST_S(ecx, ecx)")
        self.assertEqual(_make_condition("jns", "test", operands)[0],
                         "((int32_t)(ecx & ecx) >= 0)")

    def test_emitted_c_uses_test_operand_sign_bit(self):
        compiler = shutil.which("gcc") or shutil.which("clang")
        if compiler is None:
            self.skipTest("gcc or clang is required to execute lifted C")
        configurations = (
            ("cl_dl", 8, "84d1", False, False),
            ("ch_dl", 8, "84d5", False, False),
            ("cx_dx", 16, "6685d1", False, False),
            ("ecx_edx", 32, "85d1", False, False),
            ("mem_dl", 8, "8416", False, False),
            ("mem_dx", 16, "668516", False, False),
            ("mem_edx", 32, "8516", False, False),
            ("cl_cl", 8, "84c9", True, False),
            ("mem_imm8", 8, "f60680", False, True),
            ("mem_imm16", 16, "66f7060080", False, True),
        )
        consumers = (
            ("js", "7800", True), ("jns", "7900", False),
            ("sets", "0f98c0", True), ("setns", "0f99c0", False),
            ("cmovs", "0f48c3", True), ("cmovns", "0f49c3", False),
        )
        functions = []
        checks = []
        for name, bits, opcode, self_test, immediate in configurations:
            sign_bit = 1 << (bits - 1)
            values = (0, sign_bit - 1, sign_bit, (1 << bits) - 1)
            for consumer, consumer_opcode, negative in consumers:
                for deferred in (False, True):
                    function = f"check_{name}_{consumer}_{int(deferred)}"
                    # MOV preserves flags while replacing all possible inputs.
                    clobber = "b900000000ba00000000be04000000" if deferred else ""
                    raw = bytes.fromhex(opcode + clobber + consumer_opcode)
                    lifted = _lift(raw)
                    if deferred:
                        self.assertIn("_flagsnap_", lifted)
                    if consumer.startswith("j"):
                        result = f"return 0; loc_{len(raw):08X}: return 1;"
                    else:
                        result = "return eax;"
                    value = "value << 8" if name == "ch_dl" else "value"
                    functions.append(f"""
                        static unsigned {function}(uint32_t value, uint32_t mask) {{
                            ecx = {value}; edx = mask; esi = 0;
                            eax = 0; ebx = 1; memory[0] = value; memory[1] = 0;
                            {lifted}
                            {result}
                        }}
                    """)
                    mask = ("value" if self_test else
                            f"{sign_bit}u" if immediate else "mask")
                    comparison = "!=" if negative else "=="
                    checks.append(f"""
                        {{
                            const uint32_t values[] = {{{', '.join(f'{v}u' for v in values)}}};
                            for (unsigned i = 0; i < 4; ++i) {{
                                for (unsigned j = 0; j < 4; ++j) {{
                                    uint32_t value = values[i], mask = values[j];
                                    unsigned expected = ((value & {mask}) & {sign_bit}u) {comparison} 0;
                                    unsigned actual = {function}(value, mask);
                                    if (actual != expected) {{
                                        if (errors < 8) fprintf(stderr,
                                            "{function}: value=%08x mask=%08x expected=%u actual=%u\\n",
                                            (unsigned)value, (unsigned)mask, expected, actual);
                                        ++errors;
                                    }}
                                }}
                            }}
                        }}
                    """)
        source = """
            #include <stdint.h>
            #include <stdio.h>
            static uint32_t eax, ebx, ecx, edx, esi, memory[2];
            #define LO8(x) ((uint8_t)(x))
            #define HI8(x) ((uint8_t)((x) >> 8))
            #define LO16(x) ((uint16_t)(x))
            #define SET_LO8(x, v) ((x) = ((x) & 0xffffff00u) | (uint8_t)(v))
            #define MEM8(a) (*(uint8_t *)((uint8_t *)memory + (a)))
            #define MEM16(a) (*(uint16_t *)((uint8_t *)memory + (a)))
            #define MEM32(a) (*(uint32_t *)((uint8_t *)memory + (a)))
            #define TEST_S(a, b) ((int32_t)((uint32_t)(a) & (uint32_t)(b)) < 0)
        """ + "\n".join(functions) + "\nint main(void) { unsigned errors = 0;\n"
        source += "\n".join(checks)
        source += '\nif (errors) fprintf(stderr, "%u sign-condition failures\\n", errors);\n'
        source += "return errors != 0; }\n"
        with tempfile.TemporaryDirectory(prefix="test-sign-width-") as directory:
            source_path = Path(directory) / "check.c"
            executable = Path(directory) / "check.exe"
            source_path.write_text(source, encoding="utf-8")
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra",
                            str(source_path), "-o", str(executable)], check=True,
                           capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(executable)], capture_output=True,
                                    text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
