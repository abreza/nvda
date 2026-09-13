# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Optional script-based routing for unmarked Persian and English text."""

from collections.abc import Iterator
import unicodedata


def languageRuns(text: str) -> Iterator[tuple[str, str | None]]:
	"""Preserve every character, attaching numbers and punctuation to the preceding script.

	This is a Persian/English heuristic, not general language identification. Explicit
	phoneme blocks must reach the service intact and disable splitting for the segment.
	"""
	if "[[" in text:
		yield text, None
		return
	start = 0
	language = None
	for index, char in enumerate(text):
		if not unicodedata.category(char).startswith("L"):
			continue
		name = unicodedata.name(char, "")
		nextLanguage = "fa" if name.startswith("ARABIC ") else "en" if name.startswith("LATIN ") else None
		if nextLanguage is None:
			continue
		if language is not None and nextLanguage != language:
			yield text[start:index], language
			start = index
		language = nextLanguage
	if start < len(text):
		yield text[start:], language
