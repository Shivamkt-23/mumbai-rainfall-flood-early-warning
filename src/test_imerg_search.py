import earthaccess
from pathlib import Path


# =========================================================
# AUTHENTICATION
# =========================================================

print("Authenticating with NASA Earthdata...")

auth = earthaccess.login(
    strategy="environment"
)

if not auth.authenticated:
    raise RuntimeError(
        "NASA Earthdata authentication failed."
    )

print("Authentication successful.")


# =========================================================
# SEARCH ONE GRANULE
# =========================================================

print("\nSearching for one IMERG granule...")

results = earthaccess.search_data(
    concept_id="C2723754847-GES_DISC",
    temporal=(
        "2024-09-01T00:00:00",
        "2024-09-01T00:29:59",
    ),
    bounding_box=(
        72.7,
        18.8,
        73.1,
        19.3,
    ),
    count=1,
)

if not results:
    raise RuntimeError(
        "No IMERG granule found."
    )

granule = results[0]

print(
    "\nFound granule:"
)

print(granule)

print(
    "\nSize:",
    granule.size,
    "MB"
)


# =========================================================
# DOWNLOAD
# =========================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "GPM_IMERG"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "\nDownloading to:",
    OUTPUT_DIR
)

files = earthaccess.download(
    [granule],
    local_path=str(OUTPUT_DIR),
    threads=1,
)

print(
    "\nDownload complete!"
)

for file in files:
    print(
        "Downloaded:",
        file
    )