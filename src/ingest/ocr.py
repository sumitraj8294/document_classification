import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from pdf2image import convert_from_path
import cv2
import os
import tempfile

def pdf_to_images(pdf_path, out_dir=None, dpi=200):
    """
    Convert a PDF into images (one per page).

    Args:
        pdf_path (str): Path to PDF file.
        out_dir (str): Output directory for images (default: system temp dir).
        dpi (int): Resolution for rendering PDF.

    Returns:
        list[str]: Paths to generated PNG images.
    """
    out_dir = out_dir or os.path.join(tempfile.gettempdir(), "docintel_pages")
    os.makedirs(out_dir, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=dpi)
    paths = []

    for i, img in enumerate(images):
        path = os.path.join(out_dir, f"{os.path.basename(pdf_path)}_page_{i+1}.png")
        img.save(path, "PNG")
        paths.append(path)

    return paths


def preprocess_for_ocr(image_path):
    """
    Preprocess image for OCR (grayscale + denoise + threshold).

    Args:
        image_path (str): Path to the image.

    Returns:
        numpy.ndarray: Preprocessed binary image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise (Non-local Means Denoising)
    gray = cv2.fastNlMeansDenoising(gray, h=7)

    # Thresholding (Otsu’s method)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return th


def ocr_image(image_path, lang="eng"):
    """
    Perform OCR on a single image.

    Args:
        image_path (str): Path to image file.
        lang (str): OCR language (default: "eng").

    Returns:
        str: Extracted text.
    """
    proc = preprocess_for_ocr(image_path)
    return pytesseract.image_to_string(proc, lang=lang)


def extract_text_from_pdf(pdf_path, lang="eng"):
    """
    Extract text from a PDF file using OCR.

    Args:
        pdf_path (str): Path to PDF file.
        lang (str): OCR language (default: "eng").

    Returns:
        str: Extracted text from all pages.
    """
    text_parts = []
    page_imgs = pdf_to_images(pdf_path)

    for p in page_imgs:
        text_parts.append(ocr_image(p, lang=lang))

    return "\n".join(text_parts).strip()
