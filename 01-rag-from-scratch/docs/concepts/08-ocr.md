# 8. OCR: when your slides are pictures

## The real-world problem
The first real test documents for this project were three Digital Marketing lectures: 150 slides in total.
A plain PPTX text extractor found **zero words in all 150**. Every slide was a single full-slide JPEG
(slides exported as images). A naive RAG system would index them as empty and then answer "not found" to
everything, without telling you why.

This is common: slides exported as images, scanned handouts, phone photos of a whiteboard.

## What OCR does
Optical Character Recognition is two neural networks in sequence:
1. **Detection**: find the boxes in the image that contain text lines.
2. **Recognition**: read the characters inside each box.

We use **RapidOCR**, which runs PaddleOCR's detection and recognition models through ONNX Runtime:
CPU-only, pip-installable, no system packages (unlike Tesseract, which needs an OS install).

## Rebuilding the layout ([`ingest/ocr.py`](../../src/studyrag/ingest/ocr.py))
OCR returns unordered line boxes. We reconstruct:
- **Lines**: boxes whose vertical centres are within half a line-height belong to the same line, sorted left to right.
- **Reading order**: lines sorted top to bottom.
- **Paragraphs**: a vertical gap larger than 1.2× the median line height starts a new paragraph.
- **The slide title**: the tallest text line in the top 30% of the image. This becomes the chunk heading,
  so citations read "Slide 4: Online Market space".
- **Noise removal**: a lone number in the bottom corner is the slide number.

## When OCR runs ([`ingest/loaders.py`](../../src/studyrag/ingest/loaders.py))
- **PPTX**: if a slide has fewer than 8 real text words, OCR its pictures that cover at least 10% of the
  slide (logos and icons are skipped).
- **PDF**: if a page has fewer than 8 extractable words, it is probably scanned, so OCR its embedded images.
- **PNG/JPG** files: always OCR.

## Cost and caching
OCR takes about 1–3 seconds per slide on an old laptop CPU. Results are cached in `storage/ocr_cache.json`,
keyed by a SHA-1 hash of the image bytes, so re-indexing (for example after changing chunking settings)
never repeats OCR work.

## Interview questions
- *How would you know OCR quality is good enough?* Measure retrieval and answer accuracy on questions
  about OCR'd slides. That's exactly what our private evaluation set does.
- *Why hash the image bytes rather than the file name?* The same slide image can appear in several files,
  and a renamed file has the same content.
- *What about handwriting?* These models are trained on printed text; handwriting needs a different model.
