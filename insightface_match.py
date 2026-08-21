
import argparse
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np

SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SUPPORTED_PDF_EXT = {".pdf"}
MAX_DETECTION_DIM = 1024

_APP = None  # lazily-initialized InsightFace FaceAnalysis instance (cache across calls)


def get_app(det_size=(640, 640)):
    
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis
        _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=0, det_size=det_size)
    return _APP


def pdf_to_image(pdf_path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError("PDF has no pages.")
    page = doc.load_page(0)
    zoom = 300 / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    pix.save(tmp_path)
    doc.close()
    return tmp_path


def load_as_image_path(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SUPPORTED_IMAGE_EXT:
        return file_path
    if ext in SUPPORTED_PDF_EXT:
        return pdf_to_image(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


def _downscale_if_needed(image_path: str, max_dim: int = MAX_DETECTION_DIM) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image_path
    scale = max_dim / longest
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    cv2.imwrite(tmp_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return tmp_path


def get_largest_face_embedding(image_path: str, label: str) -> np.ndarray:
    """Detect all faces, keep the largest by bbox area, return its embedding."""
    working_path = _downscale_if_needed(image_path)
    try:
        img = cv2.imread(working_path)
        if img is None:
            raise ValueError(f"Could not read image for {label}: '{image_path}'")

        faces = get_app().get(img)
        if not faces:
            raise ValueError(f"No face detected in {label} image: '{image_path}'")

        def area(f):
            x1, y1, x2, y2 = f.bbox
            return (x2 - x1) * (y2 - y1)

        largest = max(faces, key=area)
        return largest.normed_embedding  # already L2-normalized
    finally:
        if working_path != image_path and os.path.exists(working_path):
            os.remove(working_path)


def _decide(similarity: float, approve_threshold: float, review_threshold: float) -> str:
   
    if similarity >= approve_threshold:
        return "approve"
    if similarity >= review_threshold:
        return "manual_review"
    return "reject"


def compare_faces(
    selfie_path: str,
    id_doc_path: str,
    approve_threshold: float = 0.55,
    review_threshold: float = 0.35,
    benchmark: bool = False,
) -> dict:
    t0 = time.perf_counter()

    selfie_img_path = load_as_image_path(selfie_path)
    id_img_path = load_as_image_path(id_doc_path)
    t1 = time.perf_counter()

    selfie_emb = get_largest_face_embedding(selfie_img_path, "selfie")
    t2 = time.perf_counter()
    id_emb = get_largest_face_embedding(id_img_path, "ID document")
    t3 = time.perf_counter()

    for p in (selfie_img_path, id_img_path):
        if p not in (selfie_path, id_doc_path) and os.path.exists(p):
            os.remove(p)

    similarity = float(np.dot(selfie_emb, id_emb))  # both already L2-normalized
    decision = _decide(similarity, approve_threshold, review_threshold)
    confidence_pct = round(max(0.0, min(1.0, similarity)) * 100, 2)

    if benchmark:
        print("\n--- Timing breakdown ---", file=sys.stderr)
        print(f"{'load/convert inputs':<28} {t1 - t0:6.2f}s", file=sys.stderr)
        print(f"{'selfie detect+embed':<28} {t2 - t1:6.2f}s", file=sys.stderr)
        print(f"{'ID detect+embed':<28} {t3 - t2:6.2f}s", file=sys.stderr)
        print(f"{'TOTAL':<28} {t3 - t0:6.2f}s", file=sys.stderr)

    return {
        "decision": decision,
        "verified": decision == "approve",
        "confidence_pct": confidence_pct,
        "cosine_similarity": round(similarity, 4),
        "approve_threshold": approve_threshold,
        "review_threshold": review_threshold,
        "model": "buffalo_l (InsightFace / ONNX Runtime)",
    }


def main():
    parser = argparse.ArgumentParser(description="Compare a selfie against an ID document photo.")
    parser.add_argument("--selfie", required=True)
    parser.add_argument("--id-doc", required=True)
    parser.add_argument("--approve-threshold", type=float, default=0.55)
    parser.add_argument("--review-threshold", type=float, default=0.35)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    try:
        result = compare_faces(
            args.selfie, args.id_doc,
            approve_threshold=args.approve_threshold,
            review_threshold=args.review_threshold,
            benchmark=args.benchmark,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()