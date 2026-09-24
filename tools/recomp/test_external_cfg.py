import unittest

from .translator import FunctionTranslator


class ExternalCfgTest(unittest.TestCase):
    def make_translator(self, expected, recovered):
        translator = FunctionTranslator.__new__(FunctionTranslator)
        translator.func_db = {
            0x1000: {
                "end": 0x1100,
                "external_cfg_instruction_count": expected,
            },
        }
        translator._recovered_cfg = {}
        translator._recover_cfg = lambda *args: recovered
        return translator

    def test_authenticated_cfg_is_retained(self):
        recovered = ([object(), object()], {0x1080: [0x1090]}, set())
        translator = self.make_translator(2, recovered)

        translator.recover_external_cfgs()

        self.assertEqual(
            translator._recovered_cfg[0x1000]["instructions"],
            recovered[0])

    def test_instruction_count_mismatch_fails_closed(self):
        translator = self.make_translator(
            3, ([object(), object()], {}, set()))

        with self.assertRaisesRegex(ValueError, "has 2 instructions"):
            translator.recover_external_cfgs()

    def test_zero_instruction_count_fails_closed(self):
        """A record that authenticates an empty body agrees with itself."""
        translator = self.make_translator(0, ([], {}, set()))

        with self.assertRaisesRegex(ValueError, "decoded no instructions"):
            translator.recover_external_cfgs()

    def test_external_entry_is_strong(self):
        translator = FunctionTranslator.__new__(FunctionTranslator)
        translator.coalesced_function_starts = set()
        self.assertTrue(translator._is_strong_entry({
            "external_entry": True,
            "has_prologue": False,
            "called_by": [],
        }))


if __name__ == "__main__":
    unittest.main()
