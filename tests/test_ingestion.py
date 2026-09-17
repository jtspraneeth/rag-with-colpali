import pytest
from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.txt_parser import TXTParser
from src.ingestion.metadata import DocumentChunk

def test_structure_aware_chunker(tmp_path):
    chunker = StructureAwareChunker(chunk_size=100, chunk_overlap=20)
    text = "SECTION I: Introduction\n\nThis is paragraph one containing factual details. This is paragraph two which goes into further explanation."
    
    chunks = chunker.split_text(
        text=text,
        document_id="doc_test_1",
        document_name="test.txt",
        page_number=1,
        source_type="txt"
    )

    assert len(chunks) > 0
    assert chunks[0].document_id == "doc_test_1"
    assert chunks[0].page_number == 1
    assert chunks[0].source_type == "txt"

def test_txt_parser(tmp_path):
    file_p = tmp_path / "sample.txt"
    file_p.write_text("Header 1\n\nThis is sample text inside the document. It contains revenue figures for FY2024.", encoding="utf-8")

    parser = TXTParser()
    chunks = parser.parse_txt(str(file_p), document_id="txt_001")

    assert len(chunks) >= 1
    assert chunks[0].document_name == "sample.txt"
    assert "revenue figures" in chunks[0].text
