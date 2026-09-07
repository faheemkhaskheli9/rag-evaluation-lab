import io

import pytest
from pypdf import PdfWriter


@pytest.fixture
def pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def text_bytes() -> bytes:
    return "The quick brown fox jumps over the lazy dog.\nSecond line.".encode("utf-8")
