import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    if not pdf_path:
        return ""

    if pdf_path.endswith('.html'):
        # It's HTML text file
        try:
            with open(pdf_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Failed to read HTML file {pdf_path}: {e}")
            return ""

    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")
        return ""