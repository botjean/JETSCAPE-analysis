"""The shared ASCII reader must also handle tiny pilot samples."""

import pytest

from jetscape_analysis.analysis.reader._parse_ascii import retrieve_lines_from_end_of_file


@pytest.mark.parametrize(
    "text,delimiter,expected",
    [
        ("", "#", ""),
        ("abc", "#", "abc"),
        ("abc\n", "\n", "abc\n"),
        ("first\nsecond\n", "\n", "second\n"),
        ("# event\nweight 1\n", "#", " event\nweight 1\n"),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 5, 1000])
def test_backward_reader_small_and_marker_free_files(tmp_path, text, delimiter, expected, chunk_size):
    path = tmp_path / "small.txt"
    path.write_text(text)
    with path.open() as stream:
        assert retrieve_lines_from_end_of_file(stream, chunk_size, delimiter) == expected


def test_backward_reader_rejects_zero_chunk(tmp_path):
    path = tmp_path / "small.txt"
    path.write_text("test")
    with path.open() as stream, pytest.raises(ValueError, match="positive"):
        retrieve_lines_from_end_of_file(stream, 0)
