"""Parse hybrid ascii input files in chunks.

.. codeauthor:: Raymond Ehlers
"""

import logging
import os
import typing
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

import attrs
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class ReachedEndOfFileException(Exception):
    """Indicates that we've somehow hit the end of the file.

    We have a separate exception so we can pass additional information
    about the context if desired.
    """


class ReachedXSecAtEndOfFileException(ReachedEndOfFileException):
    """Indicates that we've hit the cross section in the last line of the file."""


class FailedToParseHeader(Exception):
    """Indicates that we failed to parse the header."""


class LineIsNotHeader(Exception):
    """Indicates that we expected to find a header line, but we did not."""


@attrs.frozen
class CrossSection:
    value: float = attrs.field()
    error: float = attrs.field()


@attrs.frozen
class HeaderInfo:
    event_number: int = attrs.field()
    event_plane_angle: float = attrs.field()
    n_particles: int = attrs.field()
    event_weight: float = attrs.field(default=-1)
    centrality: float = attrs.field(default=-1)
    pt_hat: float = attrs.field(default=-1)
    vertex_x: float = attrs.field(default=-999)
    vertex_y: float = attrs.field(default=-999)
    vertex_z: float = attrs.field(default=-999)


def retrieve_lines_from_end_of_file(
    f: typing.TextIO, read_chunk_size: int = 100, character_to_search_for: str = "\n"
) -> str:
    """Retrieve line(s) from end of file up to a desired character.

    We grab chunks from the end of the file towards the beginning until find the character
    that we are looking for. Based on: https://stackoverflow.com/a/7167316/12907985

    Note:
        This approach requires that the file is ascii encoded. It will not work properly
        for e.g. utf-8 encoded files!

    Args:
        f: File-like object.
        read_chunk_size: Size of step in bytes to read backwards into the file. Default: 100.
        character_to_search_for: Character to search for from the back of the file. Default: `\n`,
            which will retrieve the last line of the file.

    Returns:
        The file contents from the character_to_search_for to the end of the file, assuming it's found.
    """
    if read_chunk_size <= 0:
        raise ValueError("read_chunk_size must be positive")
    last_line = ""
    while True:
        # We grab chunks from the end of the file towards the beginning until we
        # get a new line
        # However, we need to be more careful because seeking directly from the end
        # of a text file is apparently undefined behavior. To address this, we
        # follow the approach from here: https://stackoverflow.com/a/51131242/12907985
        # First, move to the end of the file.
        f.seek(0, os.SEEK_END)
        # Then, just back from the current position (based on SEEK_SET and tell()
        # of the current position).
        available = f.tell() - len(last_line)
        n_read = min(read_chunk_size, available)
        f.seek(available - n_read, os.SEEK_SET)
        # NOTE: This chunk isn't necessarily going back read_chunk_size characters, but
        #       rather just some number of bytes. Depending on the encoding, it may be.
        #       In any case, we don't care - we're just looking for the last line.
        chunk = f.read(n_read)

        if not chunk:
            # The whole file is one big line
            return last_line

        # Ignore the trailing newline at the end of the file (but include it
        # in the output).
        # NOTE: We only want to do this if we're searching for newline. It doesn't make sense
        #       to e.g. skip the last "#".
        if not last_line and character_to_search_for == "\n" and chunk.endswith("\n"):
            last_line = "\n"
            chunk = chunk[:-1]

        # NOTE: What's being searched for will have to be modified if we are dealing with
        # files with non-unix line endings.
        nl_pos = chunk.rfind(character_to_search_for)

        last_line = chunk[nl_pos + 1 :] + last_line

        if nl_pos == -1:
            # The whole chunk is part of the last line.
            continue

        return last_line


@contextmanager
def save_file_position(file_obj: typing.TextIO) -> Iterator[None]:
    """Save and restore the current position of an open file-like object.

    Args:
        file_obj: File-like object.
    Yield:
        None. The original file position is restored upon exiting the context manager.
    """
    pos = file_obj.tell()
    try:
        yield
    finally:
        file_obj.seek(pos)


class ParseCrossSection(Protocol):
    """Functional interface for parsing the cross section from a line.

    Args:
        line: Line to be parsed.
    Returns:
        Cross section extracted from the line.
    """
    def __call__(self, line: str) -> CrossSection: ...


def extract_x_sec_and_error(
    f: typing.TextIO,
    *,
    search_character_to_find_line_containing_cross_section: str,
    start_of_line_containing_cross_section: str,
    parse_cross_section_line: ParseCrossSection,
    read_chunk_size: int = 100,
) -> CrossSection | None:
    """Extract cross section and error from the end of the file.

    The most precise cross section and cross section error are available at the end of generation.
    Thus, we want to extract that value. However, the files tend to be very large, so
    we don't want to have to read the entire file just to get this value.

    Instead, we search for some key character from the end of the file, such that the
    cross section and error is between that character and the end of file.

    Note:
        The search character could go back further than the cross section line. We will then
        search for the start_of_line_containing_cross_section. e.g. for this data file:
        ```
        # Event 0
        cross_section 123.456
        ...
        ```
        We could use character="#" and start_of_line="cross_section" to retrieve the desired values.
        However, it's also okay if the two lines coincide.

    Args:
        f: File-like object pointing to our output file.
        search_character_to_find_line_containing_cross_section: Character to search for from the
            back of the file to retrieve at file contents which includes the cross section and error.
            The file contents could contain many lines.
        start_of_line_containing_cross_section: String corresponding to the start of the line
            containing the cross section and error. This DOES NOT need to be the same line as the
            search character.
        parse_cross_section_line: Function to parse the line containing the cross section.
            It should return a CrossSection.
        read_chunk_size: Size of step in bytes to read backwards into the file. Default: 100.

    Returns:
        Cross section and error, if found.
    """
    # Keep track of the incoming file position so we can reset afterwards
    with save_file_position(f):
        # Retrieve the last line of the file.
        last_event = retrieve_lines_from_end_of_file(
            f=f,
            read_chunk_size=read_chunk_size,
            character_to_search_for=search_character_to_find_line_containing_cross_section,
        )

    # logger.debug(f"last line: {last_line}")
    for line in last_event.split("\n"):
        if line.startswith(start_of_line_containing_cross_section):
            # logger.debug("Parsing xsec")
            return parse_cross_section_line(line=line)

    return None


@attrs.frozen
class ModelParameters:
    """Model dependent parameters and parsing functions.

    Attributes:
        model_name: Name of the model.
        column_names: List of column names stored in the file, ordered by their column order in the file.
        extract_x_sec_and_error: Function to extract the cross section and cross section error
            from a file-like object.
        event_by_event_generator: Generator to yield event-by-event information, switching back
            and forth both between headers and event particles.
        has_file_format_line_at_beginning_of_file: If True, the first line of the file denotes
            the file format. We need to skip the file format line when performing the full parsing
            of the file. This is needed since the file format line at the start of the file is not
            expected by the event-by-event parsing function.
    """

    model_name: str = attrs.field()
    column_names: list[str] = attrs.field()
    extract_x_sec_and_error: Callable[[typing.TextIO], CrossSection | None] = attrs.field()
    event_by_event_generator: Callable[[Iterator[str]], Iterator[HeaderInfo | str]] = (
        attrs.field()
    )
    has_file_format_line_at_beginning_of_file: bool = attrs.field()


class ChunkNotReadyException(Exception):
    """Indicate that the chunk hasn't been parsed yet, and therefore is not ready."""


@attrs.define
class ChunkGenerator:
    """Generate a chunk of the file.

    Args:
        g: Iterator over the input file.
        events_per_chunk: Number of events for the chunk.
        model_parameters: Model dependent parameters and parsing functions.
        cross_section: Cross section information.
    """

    g: Iterator[str] = attrs.field()
    _events_per_chunk: int = attrs.field()
    model_parameters: ModelParameters = attrs.field()
    cross_section: CrossSection | None = attrs.field(default=None)
    _headers: list[HeaderInfo] = attrs.Factory(list)
    _reached_end_of_file: bool = attrs.field(default=False)

    def _is_chunk_ready(self) -> bool:
        """True if the chunk is ready"""
        return not (len(self._headers) == 0 and not self._reached_end_of_file)

    def _require_chunk_ready(self) -> None:
        """Require that the chunk is ready (ie. been parsed).

        Raises:
            ChunkNotReadyException: Raised if the chunk isn't ready.
        """
        if not self._is_chunk_ready():
            raise ChunkNotReadyException()

    @property
    def events_per_chunk(self) -> int:
        return self._events_per_chunk

    @property
    def reached_end_of_file(self) -> bool:
        self._require_chunk_ready()
        return self._reached_end_of_file

    @property
    def events_contained_in_chunk(self) -> int:
        self._require_chunk_ready()
        return len(self._headers)

    @property
    def headers(self) -> list[HeaderInfo]:
        self._require_chunk_ready()
        return self._headers

    def n_particles_per_event(self) -> npt.NDArray[np.int64]:
        self._require_chunk_ready()
        return np.array([header.n_particles for header in self._headers])

    def event_split_index(self) -> npt.NDArray[np.int64]:
        self._require_chunk_ready()
        # NOTE: We skip the last header due to the way that np.split works.
        #       It will go from the last index to the end of the array.
        return np.cumsum([header.n_particles for header in self._headers])[:-1]

    @property
    def incomplete_chunk(self) -> bool:
        self._require_chunk_ready()
        return len(self._headers) != self._events_per_chunk

    def __iter__(self) -> Iterator[str]:
        # Setup parsing functions
        event_by_event_generator = self.model_parameters.event_by_event_generator

        for _ in range(self._events_per_chunk):
            # logger.debug(f"i: {i}")
            try:
                event_iter = event_by_event_generator(
                    self.g,
                )
                # NOTE: Typing gets ignored here because I don't know how to encode the additional
                #       information that the first yielded line will be the header, and then the rest
                #       will be strings. So we just ignore typing here. In principle, we could split
                #       up the _parse_event function, but I find condensing it into a single function
                #       to be more straightforward from a user perspective.
                # First, get the header. We know this first line must be a header
                self._headers.append(next(event_iter))  # type: ignore[arg-type]
                # Then we yield the rest of the particles in the event
                yield from event_iter  # type: ignore[misc]
            except (ReachedEndOfFileException, ReachedXSecAtEndOfFileException):
                # If we're reached the end of file, we should note that inside the chunk
                # because it may not have reached the full set of events per chunk.
                self._reached_end_of_file = True
                # Since we've reached this point, we need to stop.
                break
            # NOTE: If we somehow reached StopIteration, it's also fine - just
            #       allow it to propagate through and end the for loop.
