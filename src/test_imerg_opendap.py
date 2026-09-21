import earthaccess

CONCEPT_ID = "C2723754847-GES_DISC"

auth = earthaccess.login(strategy="environment")

granules = earthaccess.search_data(
    concept_id=CONCEPT_ID,
    temporal=("2024-09-25T17:00:00Z", "2024-09-25T18:00:00Z"),
    count=1
)

print("Granules found:", len(granules))

if not granules:
    raise RuntimeError("No IMERG granule found.")

granule = granules[0]

print("\nGranule:")
print(granule)

print("\nRelated URLs:")

umm = granule.get("umm", {})

for url in umm.get("RelatedUrls", []):
    print("Type:", url.get("Type"))
    print("Subtype:", url.get("Subtype"))
    print("URL:", url.get("URL"))
    print("-" * 60)