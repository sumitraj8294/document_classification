import os
from src.ingest.ocr import extract_text_from_pdf, pdf_to_images
from src.models import infer
from src.db.database import Document, Content, Metadata, SessionLocal

def summarize(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:max_chars]

def process_document(file_path: str, filename: str):
    # 1) OCR
    text = extract_text_from_pdf(file_path)

    # 2) First page image for CNN classification
    page_imgs = pdf_to_images(file_path)
    first_page_img = page_imgs[0] if page_imgs else None

    # 3) Predict doc type (CNN on image) and category (ANN on text)
    doc_type = infer.predict_doc_type(first_page_img) if first_page_img else "Other"
    category = infer.predict_text_category(text)

    # 4) Store in DB
    db = SessionLocal()
    try:
        doc = Document(filename=filename, doc_type=doc_type)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        content = Content(doc_id=doc.id, text=text, summary=summarize(text))
        meta = Metadata(doc_id=doc.id, category=category)
        db.add_all([content, meta])
        db.commit()

        result = {
            "id": doc.id,
            "filename": filename,
            "doc_type": doc_type,
            "category": category
        }
    finally:
        db.close()

    # 5) (Optional) Cleanup tmp images
    for p in page_imgs:
        try:
            os.remove(p)
        except Exception:
            pass

    return result
