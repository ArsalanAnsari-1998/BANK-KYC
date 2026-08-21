''' 
Run the below command 1st to activate the server before running app.py --
python -m uvicorn server_insightface:app --host 127.0.0.1 --port 8000

'''

import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from insightface_match import compare_faces, get_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading InsightFace model...")
    start = time.perf_counter()
    get_app()  # builds and caches the FaceAnalysis app/ONNX sessions once
    print(f"[startup] Ready in {time.perf_counter() - start:.1f}s.")
    yield


app = FastAPI(title="Face Match Verification API (InsightFace)", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "service": "Face Match Verification API",
        "model": "buffalo_l (InsightFace / ONNX Runtime)",
        "endpoints": {
            "GET /health": "liveness check",
            "POST /verify": "multipart form: 'selfie' + 'id_doc' files "
                             "(jpg/png/pdf); optional query params "
                             "approve_threshold, review_threshold, benchmark",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify")
async def verify(
    selfie: UploadFile = File(...),
    id_doc: UploadFile = File(...),
    approve_threshold: float = Query(0.55),
    review_threshold: float = Query(0.35),
    benchmark: bool = Query(False),
):
    tmp_dir = tempfile.mkdtemp(prefix="facematch_")
    selfie_path = os.path.join(tmp_dir, selfie.filename or "selfie.jpg")
    id_doc_path = os.path.join(tmp_dir, id_doc.filename or "id_doc.jpg")

    try:
        with open(selfie_path, "wb") as f:
            shutil.copyfileobj(selfie.file, f)
        with open(id_doc_path, "wb") as f:
            shutil.copyfileobj(id_doc.file, f)

        result = compare_faces(
            selfie_path, id_doc_path,
            approve_threshold=approve_threshold,
            review_threshold=review_threshold,
            benchmark=benchmark,
        )
        return JSONResponse(result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_insightface:app",
        host=os.getenv("HOST"),
        port=int(os.getenv("PORT")),
        reload=False,
    )