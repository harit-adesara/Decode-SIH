# import os
# import re
# from pathlib import Path

# import requests
# from dotenv import load_dotenv

# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     Distance,
#     VectorParams,
#     PointStruct,
# )


# # ============================================================
# # PATHS / ENV
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent.parent

# load_dotenv(BASE_DIR / ".env")

# # File containing <<<SCHEME_START:N>>> markers
# FILE_PATH = Path(__file__).resolve().parent / "Indian_Government_Health_Schemes_Complete.txt"

# COLLECTION_NAME = "health_schemes"

# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# JINA_API_KEY = os.getenv("JINA_API_KEY")

# # jina-embeddings-v3 = 1024 dimensions
# EMBEDDING_SIZE = 768


# # ============================================================
# # VALIDATE ENV
# # ============================================================

# if not QDRANT_URL:
#     raise ValueError("QDRANT_URL is missing from .env")

# if not QDRANT_API_KEY:
#     raise ValueError("QDRANT_API_KEY is missing from .env")

# if not JINA_API_KEY:
#     raise ValueError("JINA_API_KEY is missing from .env")


# # ============================================================
# # QDRANT CLIENT
# # ============================================================

# client = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY,
# )


# # ============================================================
# # CREATE / RESET COLLECTION
# # ============================================================

# def create_collection():

#     # Delete old collection because previous upload was incorrect
#     if client.collection_exists(COLLECTION_NAME):

#         print(f"Deleting old collection: {COLLECTION_NAME}")

#         client.delete_collection(
#             collection_name=COLLECTION_NAME
#         )

#     client.create_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(
#             size=EMBEDDING_SIZE,
#             distance=Distance.COSINE,
#         ),
#     )

#     print(f"Created collection: {COLLECTION_NAME}")


# # ============================================================
# # DIVIDE SCHEMES
# # ============================================================

# def divide_schemes():

#     if not FILE_PATH.exists():
#         raise FileNotFoundError(
#             f"File not found:\n{FILE_PATH}"
#         )

#     text = FILE_PATH.read_text(
#         encoding="utf-8"
#     )

#     # --------------------------------------------------------
#     # Matches:
#     #
#     # <<<SCHEME_START:1>>>
#     # ........ scheme 1 ........
#     # <<<SCHEME_END:1>>>
#     #
#     # <<<SCHEME_START:2>>>
#     # ........ scheme 2 ........
#     # <<<SCHEME_END:2>>>
#     # --------------------------------------------------------

#     pattern = re.compile(
#         r"<<<SCHEME_START:(\d+)>>>\s*"
#         r"(.*?)"
#         r"\s*<<<SCHEME_END:\1>>>",
#         re.DOTALL,
#     )

#     matches = pattern.findall(text)

#     schemes = []

#     for scheme_id, content in matches:

#         scheme_id = int(scheme_id)

#         content = content.strip()

#         # ----------------------------------------------------
#         # Extract scheme name
#         #
#         # Example:
#         # 1. AYUSHMAN BHARAT - ...
#         # ----------------------------------------------------

#         name_match = re.search(
#             r"(?m)^\s*\d+\.\s*(.+)$",
#             content,
#         )

#         if name_match:

#             scheme_name = " ".join(
#                 name_match.group(1).split()
#             )

#         else:

#             scheme_name = f"Scheme {scheme_id}"

#         schemes.append(
#             {
#                 "id": scheme_id,
#                 "name": scheme_name,
#                 "text": content,
#             }
#         )

#     return schemes


# # ============================================================
# # CREATE JINA EMBEDDING
# # ============================================================

# def create_embedding(text):

#     url = "https://api.jina.ai/v1/embeddings"

#     headers = {
#         "Authorization": f"Bearer {JINA_API_KEY}",
#         "Content-Type": "application/json",
#     }

#     payload = {
#         "model": "jina-embeddings-v3",
#         "input": [text],
#     }

#     response = requests.post(
#         url,
#         headers=headers,
#         json=payload,
#         timeout=60,
#     )

#     response.raise_for_status()

#     data = response.json()

#     embedding = data["data"][0]["embedding"]

#     # Safety check
#     if len(embedding) != EMBEDDING_SIZE:
#         raise ValueError(
#             f"Expected {EMBEDDING_SIZE} dimensions, "
#             f"got {len(embedding)}"
#         )

#     return embedding


# # ============================================================
# # UPLOAD SCHEMES TO QDRANT
# # ============================================================

# def upload_schemes(schemes):

#     points = []

#     for scheme in schemes:

#         print(
#             f"Creating embedding for "
#             f"{scheme['id']}: {scheme['name']}"
#         )

#         embedding = create_embedding(
#             scheme["text"]
#         )

#         point = PointStruct(
#             id=scheme["id"],

#             vector=embedding,

#             payload={
#                 "scheme_id": scheme["id"],
#                 "scheme_name": scheme["name"],
#                 "text": scheme["text"],
#             },
#         )

#         points.append(point)

#     # --------------------------------------------------------
#     # Upload all 28 points
#     # --------------------------------------------------------

#     print("\nUploading to Qdrant...")

#     client.upsert(
#         collection_name=COLLECTION_NAME,
#         points=points,
#     )

#     print(
#         f"Uploaded {len(points)} schemes "
#         f"to Qdrant."
#     )


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("Reading file...")
#     print(f"File: {FILE_PATH}")

#     print("\nParsing schemes...")

#     schemes = divide_schemes()

#     print(
#         f"Found {len(schemes)} schemes.\n"
#     )

#     if len(schemes) != 28:

#         raise ValueError(
#             f"Expected 28 schemes, "
#             f"but found {len(schemes)}."
#         )

#     # Print schemes
#     for scheme in schemes:

#         print(
#             f"{scheme['id']}. "
#             f"{scheme['name']}"
#         )

#     print()

#     # Create fresh collection
#     create_collection()

#     # Generate embeddings + upload
#     upload_schemes(schemes)

#     print("\nDone!")


# # ============================================================
# # ENTRY POINT
# # ============================================================

# if __name__ == "__main__":
#     main()
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


# ============================================================
# PATHS / ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# CONFIG
# ============================================================

FILE_PATH = (
    Path(__file__).resolve().parent
    / "Indian_Government_Health_Schemes_Complete.txt"
)

COLLECTION_NAME = "health_schemes"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Your current Jina response is 768 dimensions
EMBEDDING_SIZE = 1024


# ============================================================
# VALIDATE ENV
# ============================================================

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing from .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is missing from .env")

if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY is missing from .env")


# ============================================================
# QDRANT CLIENT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# ============================================================
# CREATE / RESET COLLECTION
# ============================================================

def create_collection():

    if client.collection_exists(COLLECTION_NAME):

        print(
            f"Deleting old collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: "
        f"{COLLECTION_NAME}"
    )


# ============================================================
# DIVIDE SCHEMES
# ============================================================

def divide_schemes():

    if not FILE_PATH.exists():

        raise FileNotFoundError(
            f"File not found:\n{FILE_PATH}"
        )

    text = FILE_PATH.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Expected format:
    #
    # <<<SCHEME_START:1>>>
    #
    # scheme content
    #
    # <<<SCHEME_END:1>>>
    #
    # <<<SCHEME_START:2>>>
    #
    # scheme content
    #
    # <<<SCHEME_END:2>>>
    # --------------------------------------------------------

    pattern = re.compile(
        r"<<<SCHEME_START:(\d+)>>>\s*"
        r"(.*?)"
        r"\s*<<<SCHEME_END:\1>>>",
        re.DOTALL,
    )

    matches = pattern.findall(text)

    schemes = []

    for scheme_id, content in matches:

        scheme_id = int(scheme_id)

        content = content.strip()

        # ----------------------------------------------------
        # Extract scheme name
        #
        # Example:
        #
        # 1. AYUSHMAN BHARAT - PRADHAN MANTRI...
        # ----------------------------------------------------

        name_match = re.search(
            r"(?m)^\s*\d+\.\s*(.+)$",
            content,
        )

        if name_match:

            scheme_name = " ".join(
                name_match.group(1).split()
            )

        else:

            scheme_name = (
                f"Scheme {scheme_id}"
            )

        schemes.append(
            {
                "id": scheme_id,
                "name": scheme_name,
                "text": content,
            }
        )

    return schemes


# ============================================================
# CREATE JINA EMBEDDING
# ============================================================

def create_embedding(text):

    url = "https://api.jina.ai/v1/embeddings"

    headers = {
        "Authorization": (
            f"Bearer {JINA_API_KEY}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "model": "jina-embeddings-v3",
        "input": [text],
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    embedding = data["data"][0]["embedding"]

    # --------------------------------------------------------
    # Verify dimension
    # --------------------------------------------------------

    actual_dimension = len(embedding)

    if actual_dimension != EMBEDDING_SIZE:

        raise ValueError(
            f"Embedding dimension mismatch. "
            f"Expected {EMBEDDING_SIZE}, "
            f"got {actual_dimension}"
        )

    return embedding


# ============================================================
# UPLOAD SCHEMES TO QDRANT
# ============================================================

def upload_schemes(schemes):

    points = []

    for scheme in schemes:

        print(
            f"Creating embedding for "
            f"{scheme['id']}: "
            f"{scheme['name']}"
        )

        embedding = create_embedding(
            scheme["text"]
        )

        point = PointStruct(
            id=scheme["id"],

            vector=embedding,

            payload={
                "scheme_id": scheme["id"],
                "scheme_name": scheme["name"],
                "text": scheme["text"],
            },
        )

        points.append(point)

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    print(
        "\nUploading to Qdrant..."
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"Uploaded {len(points)} schemes "
        f"to Qdrant."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Reading file...")

    print(
        f"File: {FILE_PATH}"
    )

    print(
        "\nParsing schemes..."
    )

    schemes = divide_schemes()

    print(
        f"Found {len(schemes)} schemes.\n"
    )

    # --------------------------------------------------------
    # Verify expected number
    # --------------------------------------------------------

    if len(schemes) != 28:

        raise ValueError(
            f"Expected 28 schemes, "
            f"but found {len(schemes)}."
        )

    # --------------------------------------------------------
    # Show schemes
    # --------------------------------------------------------

    for scheme in schemes:

        print(
            f"{scheme['id']}. "
            f"{scheme['name']}"
        )

    print()

    # --------------------------------------------------------
    # Create fresh collection
    # --------------------------------------------------------

    create_collection()

    # --------------------------------------------------------
    # Create embeddings + upload
    # --------------------------------------------------------

    upload_schemes(schemes)

    print(
        "\nDone!"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()