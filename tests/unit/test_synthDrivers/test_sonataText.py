# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Persian/English routing preserves the exact text supplied by NVDA."""

import unittest

from synthDrivers._sonata.text import languageRuns


class TestSonataText(unittest.TestCase):
	def test_scriptChangesPreserveJoinersDiacriticsAndNeutralCharacters(self) -> None:
		text = "۱۲۳، می‌روم به OpenAI's site؛ برمی‌گردم."
		runs = list(languageRuns(text))
		self.assertEqual(["fa", "en", "fa"], [language for _, language in runs])
		self.assertEqual(text, "".join(part for part, _ in runs))
		self.assertIn("می‌روم", runs[0][0])
		self.assertIn("OpenAI's site", runs[1][0])

	def test_emptyAndNeutralTextDoNotInventALanguage(self) -> None:
		self.assertEqual([], list(languageRuns("")))
		self.assertEqual([("123 ؟!\n", None)], list(languageRuns("123 ؟!\n")))

	def test_explicitPhonemesAreNotSplit(self) -> None:
		text = "بگو [[ t ɛ s t ]] لطفاً"
		self.assertEqual([(text, None)], list(languageRuns(text)))

	def test_latinDiacriticsStayWithEnglishRun(self) -> None:
		self.assertEqual([("café naïve", "en")], list(languageRuns("café naïve")))
