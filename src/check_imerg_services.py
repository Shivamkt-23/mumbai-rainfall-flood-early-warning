import earthaccess


print("Authenticating with NASA Earthdata...")

auth = earthaccess.login(
    strategy="environment"
)

if not auth.authenticated:
    raise RuntimeError(
        "NASA Earthdata authentication failed."
    )

print("Authentication successful.")

print("\nFinding IMERG Final collection...")

datasets = earthaccess.search_datasets(
    keyword="IMERG",
    count=20,
)

for dataset in datasets:

    summary = dataset.summary

    if summary.get("short-name") == "GPM_3IMERGHH":

        print("\n" + "=" * 70)
        print("IMERG FINAL COLLECTION FOUND")
        print("=" * 70)

        print(
            "Short name:",
            summary.get("short-name")
        )

        print(
            "Version:",
            summary.get("version")
        )

        print(
            "Concept ID:",
            summary.get("concept-id")
        )

        print("\nAvailable services:")

        services = dataset.services

        print(services)

        break

else:

    raise RuntimeError(
        "GPM_3IMERGHH collection was not found."
    )