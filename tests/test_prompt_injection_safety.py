import unittest

from multishot.prompt_injection_safety import prompt_requests_closed_eyes


class PromptInjectionSafetyTest(unittest.TestCase):
    def test_explicit_chinese_eye_closure_is_gated(self):
        self.assertTrue(prompt_requests_closed_eyes("他缓缓闭上眼睛，神情悲伤。"))
        self.assertTrue(prompt_requests_closed_eyes("人物双眼紧闭。"))
        self.assertTrue(prompt_requests_closed_eyes("她合上双眼。"))

    def test_explicit_english_eye_closure_is_gated(self):
        self.assertTrue(prompt_requests_closed_eyes("He slowly closes his eyes in grief."))
        self.assertTrue(prompt_requests_closed_eyes("Her eyes are closed."))
        self.assertTrue(prompt_requests_closed_eyes("The actor shuts both eyes."))

    def test_unrelated_eye_language_is_not_gated(self):
        self.assertFalse(prompt_requests_closed_eyes("His eyes are filled with tears."))
        self.assertFalse(prompt_requests_closed_eyes("A close-up of her eyes."))
        self.assertFalse(prompt_requests_closed_eyes("他眼里含着泪水。"))

    def test_negated_eye_closure_is_not_gated(self):
        self.assertFalse(prompt_requests_closed_eyes("He does not close his eyes."))
        self.assertFalse(prompt_requests_closed_eyes("Keep his eyes not closed."))
        self.assertFalse(prompt_requests_closed_eyes("不要闭上眼睛。"))
        self.assertFalse(prompt_requests_closed_eyes("双眼不能闭上。"))


if __name__ == "__main__":
    unittest.main()
