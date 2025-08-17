from src.ingest.pdf_loader import clean_text

def test_clean_text():
    assert clean_text("Hello\nWorld  \n") == "Hello World"
