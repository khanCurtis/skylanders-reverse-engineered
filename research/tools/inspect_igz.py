#!/usr/bin/env python3

"""
inspect_igz.py

Small research-oriented IGZ inspector for Skylanders / Vicarious Visions
Alchemy IGZ files.

This is intentionally an inspection tool, not an extractor.

Currently supports:
    - IGZ header
    - IGZ v5/v6/v7/v8/v9 descriptor tables
    - old fixups (v <= 6)
    - new fixups (v > 6)
    - TMHN texture references
    - basic igImage2 metadata discovery
    - texture payload bounds checking

The parser follows the structures observed in the public
igArchiveExtractor source used as a research reference.

Usage:

    python inspect_igz.py file.igz

Optional:

    python inspect_igz.py file.igz --fixups
    python inspect_igz.py file.igz --textures
    python inspect_igz.py file.igz --verbose
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


# Constants
IGZ_MAGIC_LE = b"IGZ\x01"
IGZ_MAGIC_BE = b"\x01ZGI"

TMHN_MAGIC = 0x544D484E
RVTB_MAGIC = 0x52565442
TMET_MAGIC = 0x544D4554
TSTR_MAGIC = 0x54535452
EXID_MAGIC = 0x45584944
EXNM_MAGIC = 0x45584E4D

# IGZ_Structure.locations from the research reference.
#
# locations[version]:
#     [0] attributes
#     [1] descriptor table
#     [2] old fixup/tag start
#     [3] texture metadata location
#
IGZ_LOCATIONS: dict[int, tuple[int, int, int, int]] = {
    5: (
        0x0000056C,
        0x00000010,
        0x0000001C,
        0x00000000,
    ),
    6: (
        0x0000056C,
        0x00000010,
        0x0000001C,
        0x000000A0,
    ),
    7: (
        0x0000056C,
        0x00000018,
        0x00000000,
        0x00000080,
    ),
    8: (
        0x00000224,
        0x00000018,
        0x00000000,
        0x000000B0,
    ),
    9: (
        0x0000056C,
        0x00000018,
        0x00000000,
        0x0000006C,
    ),
}


# Data classes
@dataclass
class Descriptor:
    index: int
    offset: int
    size: int
    unknown1: int
    unknown2: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass
class Fixup:
    magic: int
    offset: int
    count: int
    length: int
    start_of_data: int
    kind: str

    @property
    def data_offset(self) -> int:
        return self.offset + self.start_of_data


@dataclass
class TMHNEntry:
    index: int
    raw_size: int
    size: int
    raw_offset: int
    descriptor_index: int
    relative_offset: int
    absolute_offset: int | None


@dataclass
class ImageInfo:
    object_offset: int
    width: int
    height: int
    depth: int
    mipmaps: int
    array: int
    format_hash: int | None
    tmhn_index: int | None


# Reader
class Reader:
    """
    Minimal endian-aware reader.

    IGZ files can be represented with either endian ordering. The magic
    determines which byte order the source parser selects.
    """

    def __init__(self, data: bytes, endian: str):
        self.data = data
        self.endian = endian

    def u16(self, offset: int) -> int:
        return struct.unpack_from(self.endian + "H", self.data, offset)[0]

    def u32(self, offset: int) -> int:
        return struct.unpack_from(self.endian + "I", self.data, offset)[0]

    def bytes(self, offset: int, size: int) -> bytes:
        return self.data[offset:offset + size]

    def check(self, offset: int, size: int = 1) -> None:
        if offset < 0 or offset + size > len(self.data):
            raise ValueError(
                f"read outside file: offset=0x{offset:x}, "
                f"size=0x{size:x}, file=0x{len(self.data):x}"
            )


# Utility functions
def magic_name(magic: int) -> str:
    names = {
        TSTR_MAGIC: "TSTR",
        TMET_MAGIC: "TMET",
        EXID_MAGIC: "EXID",
        EXNM_MAGIC: "EXNM",
        TMHN_MAGIC: "TMHN",
        RVTB_MAGIC: "RVTB",
    }

    if magic in names:
        return names[magic]

    try:
        raw = struct.pack(">I", magic)
        if all(32 <= b < 127 for b in raw):
            return raw.decode("ascii")
    except Exception:
        pass

    return f"0x{magic:08X}"


def format_hex(value: int | None, width: int = 8) -> str:
    if value is None:
        return "N/A"
    return f"0x{value:0{width}X}"


def safe_u32(reader: Reader, offset: int) -> int | None:
    if offset < 0 or offset + 4 > len(reader.data):
        return None
    return reader.u32(offset)


def is_plausible_offset(value: int, file_size: int) -> bool:
    return 0 <= value < file_size


# IGZ parsing
class IGZFile:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()

        if len(self.data) < 0x0C:
            raise ValueError("File is too small to be an IGZ file.")

        self.raw_magic = self.data[:4]

        if self.raw_magic == IGZ_MAGIC_LE:
            # This matches the source parser's selection:
            #
            #   IGZ\x01 -> StreamHelper.Endianness.Little
            #
            # Note:
            # StreamHelper's naming/implementation should be treated as
            # authoritative when comparing against the original parser.
            self.endian = ">"
            self.endian_name = "big"
        elif self.raw_magic == IGZ_MAGIC_BE:
            self.endian = "<"
            self.endian_name = "little"
        else:
            raise ValueError(
                f"Not an IGZ file: magic={self.raw_magic.hex(' ')}"
            )

        self.reader = Reader(self.data, self.endian)

        self.version = self.reader.u32(0x04)
        self.crc = self.reader.u32(0x08)

        self.descriptors: list[Descriptor] = []
        self.fixups: list[Fixup] = []
        self.tmhn_entries: list[TMHNEntry] = []
        self.images: list[ImageInfo] = []

    # Header
    def parse_descriptors(self) -> None:
        if self.version not in IGZ_LOCATIONS:
            raise ValueError(
                f"Unsupported IGZ structure version: {self.version}"
            )

        descriptor_location = IGZ_LOCATIONS[self.version][1]

        pos = descriptor_location

        while True:
            self.reader.check(pos, 0x10)

            offset = self.reader.u32(pos)
            size = self.reader.u32(pos + 0x04)
            unknown1 = self.reader.u32(pos + 0x08)
            unknown2 = self.reader.u32(pos + 0x0C)

            if offset == 0:
                break

            descriptor = Descriptor(
                index=len(self.descriptors),
                offset=offset,
                size=size,
                unknown1=unknown1,
                unknown2=unknown2,
            )

            self.descriptors.append(descriptor)
            pos += 0x10

            if len(self.descriptors) > 1024:
                raise ValueError(
                    "Descriptor table exceeded 1024 entries; "
                    "file may be malformed or endian interpretation "
                    "may be incorrect."
                )

    # Fixups
    def parse_fixups(self) -> None:
        if not self.descriptors:
            raise ValueError("Descriptors must be parsed first.")

        if self.version <= 6:
            self.parse_old_fixups()
        else:
            self.parse_new_fixups()

    def parse_old_fixups(self) -> None:
        """
        Parse the v5/v6 legacy fixup table.

        For old IGZ files, the first DWORD at each fixup location is not
        the four-character fixup magic. It is a numeric fixup ID.

        The IDs observed/handled by igArchiveExtractor include:

            0x00 -> TMET
            0x01 -> TSTR
            0x02 -> EXID
            0x03 -> EXNM
            0x05 -> RVTB
            0x0A -> TMHN
            0x0C -> MTSZ
            0x0E -> RSTR

        IGZ_Fixup.Process() then parses the common fixup header.
        """

        if not self.descriptors:
            raise ValueError("Descriptors must be parsed first.")

        descriptor0 = self.descriptors[0]

        # Matches:
        #
        # uint bytesPassed = IGZ_Structure.locations[version][0x02];
        #
        bytes_passed = IGZ_LOCATIONS[self.version][2]

        # Matches:
        #
        # uint numberOfFixups =
        #     ebr.ReadUInt32WithOffset(descriptors[0].offset + 0x10);
        #
        number_of_fixups = self.reader.u32(descriptor0.offset + 0x10)

        old_fixup_names = {
            0x00: "TMET",
            0x01: "TSTR",
            0x02: "EXID",
            0x03: "EXNM",
            0x05: "RVTB",
            0x0A: "TMHN",
            0x0C: "MTSZ",
            0x0E: "RSTR",
        }

        for index in range(number_of_fixups):
            pos = descriptor0.offset + bytes_passed

            self.reader.check(pos, 4)

            fixup_id = self.reader.u32(pos)

            # IGZ_Fixup.Process():
            #
            #   sh.BaseStream.Seek(-4, Current)
            #   magicNumber = sh.ReadUInt32()
            #   offset = Position - 4
            #
            # For the old format, the ID is consumed as part of the
            # specialized fixup's common header.
            #
            # The next fields are:
            #
            #   +0x04 unknown / reserved
            #   +0x08 unknown / reserved
            #   +0x0C count
            #   +0x10 length
            #   +0x14 startOfData
            #
            self.reader.check(pos, 0x18)

            count = self.reader.u32(pos + 0x0C)
            length = self.reader.u32(pos + 0x10)
            start_of_data = self.reader.u32(pos + 0x14)

            if length == 0:
                raise ValueError(
                    f"Zero-length old fixup #{index} "
                    f"at 0x{pos:X}"
                )

            if pos + length > len(self.data):
                raise ValueError(
                    f"Old fixup #{index} at 0x{pos:X} "
                    f"extends outside the file: "
                    f"length=0x{length:X}"
                )

            # Convert the old numeric ID into the real four-character
            # magic used by the parsed fixup object.
            magic_map = {
                0x00: TMET_MAGIC,
                0x01: TSTR_MAGIC,
                0x02: EXID_MAGIC,
                0x03: EXNM_MAGIC,
                0x05: RVTB_MAGIC,
                0x0A: TMHN_MAGIC,
            }

            magic = magic_map.get(fixup_id, fixup_id)

            fixup = Fixup(
                magic=magic,
                offset=pos,
                count=count,
                length=length,
                start_of_data=start_of_data,
                kind=f"old:{old_fixup_names.get(fixup_id, 'UNKNOWN')}",
            )

            self.fixups.append(fixup)

            # Exactly matches:
            #
            # bytesPassed += fixups.Last().length;
            #
            bytes_passed += length

        if index + 1 != number_of_fixups:
            raise ValueError(
                "Old fixup count did not complete as expected."
            )

    def parse_new_fixups(self) -> None:
        """
        Reproduce the general iteration used by ReadNewFixups().

        New fixups live in descriptor 0.
        """

        descriptor = self.descriptors[0]

        pos = descriptor.offset
        end = descriptor.end

        while pos < end:
            self.reader.check(pos, 0x18)

            magic = self.reader.u32(pos)

            count = self.reader.u32(pos + 0x0C)
            length = self.reader.u32(pos + 0x10)
            start_of_data = self.reader.u32(pos + 0x14)

            if length == 0:
                raise ValueError(
                    f"Zero-length new fixup at 0x{pos:X}"
                )

            if pos + length > len(self.data):
                raise ValueError(
                    f"Fixup at 0x{pos:X} extends outside the file: "
                    f"length=0x{length:X}"
                )

            fixup = Fixup(
                magic=magic,
                offset=pos,
                count=count,
                length=length,
                start_of_data=start_of_data,
                kind="new",
            )

            self.fixups.append(fixup)

            pos += length

    # TMHN
    def parse_tmhn(self) -> None:
        """
        Parse TMHN entries and resolve their descriptor-relative offsets.
        """

        for fixup in self.fixups:
            if fixup.magic != TMHN_MAGIC:
                continue

            data_offset = fixup.data_offset

            for i in range(fixup.count):
                entry_offset = data_offset + i * 8
                self.reader.check(entry_offset, 8)

                raw_size = self.reader.u32(entry_offset)
                raw_offset = self.reader.u32(entry_offset + 4)

                size = raw_size & 0x07FFFFFF

                if self.version <= 6:
                    descriptor_index = (raw_offset >> 24) + 1
                    relative_offset = raw_offset & 0x00FFFFFF
                else:
                    descriptor_index = (raw_offset >> 27) + 1
                    relative_offset = raw_offset & 0x07FFFFFF

                absolute_offset: int | None

                if descriptor_index >= len(self.descriptors):
                    absolute_offset = None
                else:
                    absolute_offset = (
                        self.descriptors[descriptor_index].offset
                        + relative_offset
                    )

                self.tmhn_entries.append(
                    TMHNEntry(
                        index=i,
                        raw_size=raw_size,
                        size=size,
                        raw_offset=raw_offset,
                        descriptor_index=descriptor_index,
                        relative_offset=relative_offset,
                        absolute_offset=absolute_offset,
                    )
                )

    # Research dumps
    def dump_fixup_data(self, fixup_name: str) -> None:
        """
        Hex/word dump the data region belonging to a named fixup.

        This is intentionally a research aid. It does not interpret the
        bytes beyond the fixup header itself.
        """
        target = next(
            (f for f in self.fixups if magic_name(f.magic) == fixup_name),
            None,
        )

        if target is None:
            print(f"\n{fixup_name}")
            print("─" * 64)
            print("Fixup not found.")
            return

        start = target.data_offset
        end = start + target.length - target.start_of_data

        print(f"\n{fixup_name} Raw Data")
        print("─" * 64)
        print(
            f"fixup:  0x{target.offset:X} "
            f"count={target.count} "
            f"length=0x{target.length:X}"
        )
        print(
            f"data:   0x{start:X}..0x{end:X} "
            f"({end - start} bytes)"
        )

        for offset in range(start, end, 16):
            chunk = self.data[offset:min(offset + 16, end)]

            hex_part = " ".join(f"{b:02X}" for b in chunk)
            hex_part = f"{hex_part:<47}"

            ascii_part = "".join(
                chr(b) if 32 <= b < 127 else "."
                for b in chunk
            )

            print(
                f"{offset:08X}  {hex_part}  |{ascii_part}|"
            )

        print("\n32-bit words:")
        for offset in range(start, end - 3, 4):
            value = self.reader.u32(offset)
            print(f"  0x{offset:08X}: 0x{value:08X}")

    # Basic igImage2 discovery
    def parse_images(self) -> None:
        """
        Locate igImage2 objects using the TMET/RVTB relationship used by
        the research reference.

        This intentionally implements only the fields needed for texture
        research.
        """

        tmet = next(
            (f for f in self.fixups if f.magic == TMET_MAGIC),
            None,
        )

        rvtb = next(
            (f for f in self.fixups if f.magic == RVTB_MAGIC),
            None,
        )

        if tmet is None or rvtb is None:
            return

        # The complete RVTB decoding is more involved than we need here.
        # We therefore use the known v6 texture metadata location as an
        # additional inspection point.
        if self.version not in IGZ_LOCATIONS:
            return

        metadata_location = IGZ_LOCATIONS[self.version][3]

        if len(self.descriptors) < 2:
            return

        object_base = self.descriptors[1].offset + metadata_location

        # The exact object representation varies with IGZ structure.
        # We do not invent fields when the available structure is
        # insufficient. The main authoritative texture references remain
        # TMHN entries.
        if object_base >= len(self.data):
            return

        # Keep this method intentionally conservative.
        #
        # Texture metadata can be parsed once an igImage2 object offset is
        # identified through the object list. That implementation is not
        # duplicated here yet because this tool is intended to avoid
        # silently guessing at object layouts.
        return

    # Full parse
    def parse(self) -> None:
        """
        Parse the IGZ structure in dependency order.

        Descriptors must exist before fixups can be interpreted.
        TMHN must exist before texture references can be displayed.
        """

        self.parse_descriptors()
        self.parse_fixups()
        self.parse_tmhn()
        self.parse_images()

    # Output
    def print_header(self) -> None:
        print("IGZ")
        print("─" * 64)
        print(f"File:       {self.path}")
        print(f"Size:       0x{len(self.data):X} ({len(self.data):,} bytes)")
        print(f"Magic:      {self.raw_magic!r}")
        print(f"Version:    0x{self.version:X}")
        print(f"CRC:        0x{self.crc:08X}")
        print(f"Byte order: {self.endian_name}")

    def print_descriptors(self) -> None:
        print("\nDescriptors")
        print("─" * 64)

        for d in self.descriptors:
            print(
                f"[{d.index:2}] "
                f"offset={format_hex(d.offset)} "
                f"size={format_hex(d.size)} "
                f"end={format_hex(d.end)} "
                f"u1={format_hex(d.unknown1)} "
                f"u2={format_hex(d.unknown2)}"
            )

    def print_fixups(self) -> None:
        print("\nFixups")
        print("─" * 64)

        for f in self.fixups:
            print(
                f"{magic_name(f.magic):8} "
                f"offset={format_hex(f.offset)} "
                f"count={f.count:<6} "
                f"length={format_hex(f.length)} "
                f"data={format_hex(f.data_offset)} "
                f"type={f.kind}"
            )

    def print_tmhn(self) -> None:
        print("\nTMHN Texture References")
        print("─" * 64)

        if not self.tmhn_entries:
            print("No TMHN entries found.")
            return

        for entry in self.tmhn_entries:
            absolute = entry.absolute_offset

            bounds = "INVALID"

            if absolute is not None:
                end = absolute + entry.size

                if absolute >= 0 and end <= len(self.data):
                    bounds = (
                        f"OK "
                        f"[0x{absolute:X}..0x{end:X})"
                    )
                else:
                    bounds = (
                        f"OUT-OF-BOUNDS "
                        f"[0x{absolute:X}..0x{end:X})"
                    )

            print(
                f"[{entry.index:3}] "
                f"size={format_hex(entry.size)} "
                f"raw_size={format_hex(entry.raw_size)} "
                f"raw_offset={format_hex(entry.raw_offset)} "
                f"desc={entry.descriptor_index:<3} "
                f"rel={format_hex(entry.relative_offset, 6)} "
                f"absolute="
                f"{format_hex(absolute)} "
                f"{bounds}"
            )

    def print_images(self) -> None:
        print("\nigImage2")
        print("─" * 64)

        if not self.images:
            print(
                "No igImage2 objects decoded yet. "
                "TMHN references above are still authoritative "
                "for locating texture payloads."
            )
            return

        for i, image in enumerate(self.images):
            print(f"[{i}] object=0x{image.object_offset:X}")
            print(f"    width:       {image.width}")
            print(f"    height:      {image.height}")
            print(f"    depth:       {image.depth}")
            print(f"    mipmaps:     {image.mipmaps}")
            print(f"    array:       {image.array}")
            print(f"    format hash: {format_hex(image.format_hash)}")
            print(f"    TMHN index:  {image.tmhn_index}")

    def print_texture_summary(self) -> None:
        print("\nTexture Payload Summary")
        print("─" * 64)

        if not self.tmhn_entries:
            print("No TMHN entries found.")
            return

        valid = 0

        for entry in self.tmhn_entries:
            if entry.absolute_offset is None:
                continue

            end = entry.absolute_offset + entry.size

            if end <= len(self.data):
                valid += 1

        print(f"TMHN entries:       {len(self.tmhn_entries)}")
        print(f"Valid file ranges:  {valid}")

        # Highlight the observed payload boundary from the current
        # research sample without treating it as a universal IGZ rule.
        candidate = 0x1A4C

        matches = [
            e for e in self.tmhn_entries
            if e.absolute_offset == candidate
        ]

        if matches:
            print(
                f"\nObserved sample boundary 0x{candidate:X}:"
            )

            for e in matches:
                print(
                    f"  TMHN[{e.index}] "
                    f"size=0x{e.size:X} "
                    f"end=0x{candidate + e.size:X}"
                )
        else:
            print(
                f"\nNo TMHN entry begins at the observed "
                f"sample boundary 0x{candidate:X}."
            )


# CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Vicarious Visions Alchemy IGZ files."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to an .igz file",
    )

    parser.add_argument(
        "--fixups",
        action="store_true",
        help="Print parsed fixups",
    )

    parser.add_argument(
        "--textures",
        action="store_true",
        help="Print TMHN texture references",
    )

    parser.add_argument(
        "--dump-fixup",
        action="append",
        choices=["TMET", "RVTB", "EXID", "TMHN"],
        help="Dump raw data for a fixup type; may be repeated.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all available inspection sections",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.file.exists():
        print(
            f"error: file does not exist: {args.file}",
            file=sys.stderr,
        )
        return 1

    if not args.file.is_file():
        print(
            f"error: not a regular file: {args.file}",
            file=sys.stderr,
        )
        return 1

    try:
        igz = IGZFile(args.file)
        igz.parse()

        igz.print_header()
        igz.print_descriptors()

        if args.dump_fixup:
            for fixup_name in args.dump_fixup:
                igz.dump_fixup_data(fixup_name)

        if args.fixups or args.verbose:
            igz.print_fixups()

        if args.textures or args.verbose:
            igz.print_tmhn()
            igz.print_texture_summary()

        if args.verbose:
            igz.print_images()

        if not args.fixups and not args.textures and not args.verbose:
            igz.print_fixups()
            igz.print_tmhn()
            igz.print_texture_summary()

        return 0

    except Exception as exc:
        print(
            f"\nerror: {exc}",
            file=sys.stderr,
        )

        if args.verbose:
            raise

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
