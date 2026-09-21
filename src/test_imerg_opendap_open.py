import earthaccess
import xarray as xr

CONCEPT_ID = "C2723754847-GES_DISC"

earthaccess.login(strategy="environment")

granules = earthaccess.search_data(
    concept_id=CONCEPT_ID,
    temporal=("2024-09-25T17:00:00Z", "2024-09-25T18:00:00Z"),
    count=1
)

if not granules:
    raise RuntimeError("No IMERG granule found.")

granule = granules[0]

opendap_url = None

for link in granule.get("umm", {}).get("RelatedUrls", []):
    if (
        link.get("Type") == "USE SERVICE API"
        and link.get("Subtype") == "OPENDAP DATA"
    ):
        opendap_url = link["URL"]
        break

if opendap_url is None:
    raise RuntimeError("No OPeNDAP URL found.")

print("OPeNDAP URL:")
print(opendap_url)

session = earthaccess.get_requests_https_session()

print("\nOpening dataset...")

ds = xr.open_dataset(
    opendap_url,
    engine="pydap",
    session=session
)

print("\nDataset opened successfully.")
print(ds)

print("\nVariables:")
print(list(ds.data_vars))

print("\nCoordinates:")
print(list(ds.coords))