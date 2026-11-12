# SPDX-License-Identifier: AGPL-3.0-or-later
"""UCP4 record framing for the console WebSocket (:7442 and the nl-search socket).

Each WebSocket frame holds records: an 8-byte header (type, format, compressed, reserved,
u32 big-endian length) then the payload, JSON or raw bytes. A message is a HEAD record
(action, id, type) followed by a BODY record. Split out of ucp4_client.py; the framing is
what the live console has accepted since the first adoption (Protect 7.3.70).
"""
import json
import logging
import struct

log = logging.getLogger("ucp4-client")

# UCP4 record header (8 bytes): type, format, compressed, reserved, u32 length.
RECORD_TYPE_HEAD = 1
RECORD_TYPE_BODY = 2
RECORD_FORMAT_JSON = 1
RECORD_FORMAT_RAW = 3


def pack_record(record_type: int, payload) -> bytes:
    if isinstance(payload, (bytes, bytearray)):
        body = bytes(payload)
        rec_format = RECORD_FORMAT_RAW
    else:
        body = json.dumps(payload).encode()
        rec_format = RECORD_FORMAT_JSON
    header = struct.pack(">BBBBI", record_type, rec_format, 0, 0, len(body))
    return header + body


def pack_message(head: dict, body) -> bytes:
    return pack_record(RECORD_TYPE_HEAD, head) + pack_record(RECORD_TYPE_BODY, body)


def unpack_all_records(data: bytes):
    """[(record_type, format, obj)] for every complete record in a frame."""
    records = []
    off = 0
    while off < len(data):
        if len(data) - off < 8:
            log.warning("trailing %d bytes, not enough for a record header", len(data) - off)
            break
        rtype, rec_format, _compressed, _reserved, length = struct.unpack(
            ">BBBBI", data[off:off + 8])
        off += 8
        payload = data[off:off + length]
        off += length
        if rec_format == RECORD_FORMAT_RAW:
            obj = {"_raw_bytes": payload}
        else:
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                obj = {"_raw": payload.decode(errors="replace")}
        records.append((rtype, rec_format, obj))
    return records
