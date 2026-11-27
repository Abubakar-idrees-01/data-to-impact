"""Splitting the raw files into stories: a story cut off by the partial download must be dropped."""

from tinygpt.data import stories
from tinygpt.tokenizer import EOT


def test_cut_off_last_story_is_dropped(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(f"One.\n{EOT}\nTwo.\n{EOT}\n{EOT}\nThree is cut o", encoding="utf-8")
    assert stories(p) == ["One.", "Two."]               # empty story skipped, cut-off one dropped
    p.write_text(f"One.\n{EOT}\nTwo.\n{EOT}\n", encoding="utf-8")
    assert stories(p) == ["One.", "Two."]
