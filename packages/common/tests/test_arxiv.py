"""Unit tests for the arXiv metadata fetcher (research-kb#20).

Network is monkeypatched — these are pure unit tests, no live arXiv calls.
"""

import urllib.error
import urllib.request

import pytest

from research_kb_common.arxiv import arxiv_id_from_filename, fetch_arxiv_metadata

pytestmark = pytest.mark.unit


_ENTRY = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1312.5602v1</id>
    <published>2013-12-19T19:18:21Z</published>
    <title>Playing Atari with Deep Reinforcement
Learning</title>
    <author><name>Volodymyr Mnih</name></author>
    <author><name>Koray Kavukcuoglu</name></author>
  </entry>
</feed>"""

_ERROR = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
    <title>Error</title>
  </entry>
</feed>"""


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, body: bytes | None = None, exc: Exception | None = None):
    def fake_urlopen(request, timeout=None):
        if exc is not None:
            raise exc
        return _FakeResp(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


class TestArxivIdFromFilename:
    def test_plain_id(self):
        assert arxiv_id_from_filename("1312.5602.pdf") == "1312.5602"

    def test_versioned_id(self):
        assert arxiv_id_from_filename("1312.5602v2.pdf") == "1312.5602"

    def test_five_digit_suffix(self):
        assert arxiv_id_from_filename("2502.13187.pdf") == "2502.13187"

    def test_non_arxiv_filename(self):
        assert arxiv_id_from_filename("Kalman_Optimal_Control_1960.pdf") is None

    def test_not_a_pdf(self):
        assert arxiv_id_from_filename("1312.5602.txt") is None


class TestFetchArxivMetadata:
    def test_parses_entry(self, monkeypatch):
        _patch_urlopen(monkeypatch, body=_ENTRY)
        meta = fetch_arxiv_metadata("1312.5602")
        assert meta == {
            "title": "Playing Atari with Deep Reinforcement Learning",
            "authors": ["Volodymyr Mnih", "Koray Kavukcuoglu"],
            "year": 2013,
        }

    def test_error_entry_returns_none(self, monkeypatch):
        # Valid-format but unknown id → arXiv returns the /api/errors sentinel.
        _patch_urlopen(monkeypatch, body=_ERROR)
        assert fetch_arxiv_metadata("9999.99999") is None

    def test_network_error_returns_none(self, monkeypatch):
        _patch_urlopen(monkeypatch, exc=urllib.error.URLError("boom"))
        assert fetch_arxiv_metadata("1312.5602") is None

    def test_malformed_xml_returns_none(self, monkeypatch):
        _patch_urlopen(monkeypatch, body=b"<not-valid-xml")
        assert fetch_arxiv_metadata("1312.5602") is None

    def test_malformed_id_returns_none_without_request(self, monkeypatch):
        # A malformed id (e.g. comma-joined) is rejected before any network call.
        def boom(*args, **kwargs):
            raise AssertionError("must not hit the network for a malformed id")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        assert fetch_arxiv_metadata("1312.5602,2001.00001") is None
        assert fetch_arxiv_metadata("not-an-id") is None

    def test_id_mismatch_returns_none(self, monkeypatch):
        # Feed describes 1312.5602 but we asked for a different id → reject.
        _patch_urlopen(monkeypatch, body=_ENTRY)
        assert fetch_arxiv_metadata("2001.00001") is None
