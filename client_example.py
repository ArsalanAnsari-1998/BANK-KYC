"""
client_example.py
------------------
Example of calling the running face-match server. The server must already
be up (see server.py docstring). Each call here should take roughly a
second or two, NOT 2+ minutes, since models are already loaded in the
server process.

Usage:
    python client_example.py --selfie Selfie1.png --id-doc id2.jpg
"""

import argparse
import json
import time

import requests

SERVER_URL = "http://127.0.0.1:8000/verify"


def verify(selfie_path: str, id_doc_path: str, **params) -> dict:
    with open(selfie_path, "rb") as selfie_f, open(id_doc_path, "rb") as id_f:
        files = {
            "selfie": (selfie_path, selfie_f, "application/octet-stream"),
            "id_doc": (id_doc_path, id_f, "application/octet-stream"),
        }
        start = time.perf_counter()
        resp = requests.post(SERVER_URL, files=files, params=params)
        elapsed = time.perf_counter() - start

    resp.raise_for_status()
    result = resp.json()
    result["_client_measured_seconds"] = round(elapsed, 2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfie", required=True)
    parser.add_argument("--id-doc", required=True)
    args = parser.parse_args()

    result = verify(args.selfie, args.id_doc, benchmark=True)
    print(json.dumps(result, indent=2))
