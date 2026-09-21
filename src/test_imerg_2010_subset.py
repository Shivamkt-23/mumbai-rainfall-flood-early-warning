# ============================================================
# NASA IMERG HARMONY SUBSET TEST
#
# PURPOSE
# -------
# Test NASA GPM IMERG remote spatial + temporal subsetting
# for Mumbai for January 2010.
#
# IMPORTANT
# ---------
# This file is only a TEST.
#
# Once the January 2010 request works, we will reuse the same
# pipeline for the full 2010-2025 IMERG extraction.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
from datetime import datetime
from pathlib import Path

import earthaccess

from harmony import (
    Client,
    Collection,
    Request,
    BBox,
)


# ============================================================
# CONFIGURATION
# ============================================================

SHORT_NAME = "GPM_3IMERGHH"

VERSION = "07"


# ============================================================
# MUMBAI BOUNDING BOX
#
# Format:
#   west, south, east, north
# ============================================================

MUMBAI_BBOX = BBox(
    72.7,   # west
    18.9,   # south
    73.1,   # east
    19.2,   # north
)


# ============================================================
# TEST PERIOD
#
# January 2010
# ============================================================

START_TIME = datetime(
    2010,
    1,
    1,
    0,
    0,
)

END_TIME = datetime(
    2010,
    2,
    1,
    0,
    0,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "imerg_subsets"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPER — SECTION HEADER
# ============================================================

def print_section(title: str):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    print_section(
        "NASA IMERG HARMONY SUBSET TEST"
    )


    print(
        "\nTest period:"
    )

    print(
        f"{START_TIME} → {END_TIME}"
    )


    print(
        "\nMumbai bounding box:"
    )

    print(
        "West : 72.7"
    )

    print(
        "South: 18.9"
    )

    print(
        "East : 73.1"
    )

    print(
        "North: 19.2"
    )


    # ========================================================
    # STEP 1 — CHECK ENVIRONMENT TOKEN
    # ========================================================

    print_section(
        "STEP 1 — EARTHDATA TOKEN"
    )


    print(
        "\nChecking EARTHDATA_TOKEN..."
    )


    token_from_environment = os.environ.get(
        "EARTHDATA_TOKEN"
    )


    if not token_from_environment:

        raise RuntimeError(
            "\nEARTHDATA_TOKEN is NOT set.\n\n"
            "Set it in the same PowerShell window with:\n\n"
            '$env:EARTHDATA_TOKEN="YOUR_TOKEN"\n'
        )


    print(
        "EARTHDATA_TOKEN is present."
    )


    # Do NOT print the actual token.
    print(
        "Token value is hidden for security."
    )


    # ========================================================
    # STEP 2 — EARTHDATA LOGIN
    # ========================================================

    print_section(
        "STEP 2 — EARTHDATA AUTHENTICATION"
    )


    print(
        "\nAuthenticating using the environment token..."
    )


    try:

        auth = earthaccess.login(
            strategy="environment"
        )

    except Exception as error:

        print(
            "\nEarthdata authentication failed."
        )

        print(
            f"Error: {error}"
        )

        raise


    if not auth.authenticated:

        raise RuntimeError(
            "Earthdata authentication returned "
            "an unauthenticated session."
        )


    print(
        "Earthdata authentication successful."
    )


    # ========================================================
    # STEP 3 — GET ACTIVE EDL TOKEN
    # ========================================================

    print_section(
        "STEP 3 — GETTING EDL TOKEN"
    )


    try:

        edl_token = (
            earthaccess.get_edl_token()
        )

    except Exception as error:

        print(
            "\nCould not retrieve the active "
            "Earthdata token."
        )

        print(
            f"Error: {error}"
        )

        raise


    if not edl_token:

        raise RuntimeError(
            "Earthdata returned an empty token."
        )


    print(
        "Active EDL token obtained."
    )

    print(
        "Token value hidden."
    )


    # ========================================================
    # STEP 4 — SEARCH IMERG COLLECTION
    # ========================================================

    print_section(
        "STEP 4 — SEARCHING IMERG COLLECTION"
    )


    print(
        "\nShort name:"
    )

    print(
        SHORT_NAME
    )


    print(
        "\nVersion:"
    )

    print(
        VERSION
    )


    try:

        datasets = (
            earthaccess.search_datasets(
                short_name=SHORT_NAME,
                version=VERSION,
            )
        )

    except Exception as error:

        print(
            "\nFailed to search NASA collections."
        )

        print(
            f"Error: {error}"
        )

        raise


    if not datasets:

        raise RuntimeError(
            "No IMERG dataset was found."
        )


    dataset = datasets[0]


    # earthaccess 0.19+:
    # concept_id is a property, not a method.
    concept_id = dataset.concept_id


    print(
        "\nCollection:"
    )

    print(
        SHORT_NAME
    )


    print(
        "Version:"
    )

    print(
        VERSION
    )


    print(
        "Concept ID:"
    )

    print(
        concept_id
    )


    # ========================================================
    # STEP 5 — CREATE HARMONY CLIENT
    # ========================================================

    print_section(
        "STEP 5 — HARMONY CLIENT"
    )


    print(
        "\nCreating Harmony client..."
    )


    try:

        harmony_client = Client(
            token=edl_token
        )

    except Exception as error:

        print(
            "\nFailed to create Harmony client."
        )

        print(
            f"Error: {error}"
        )

        raise


    print(
        "Harmony client created successfully."
    )


    # ========================================================
    # STEP 6 — BUILD HARMONY REQUEST
    # ========================================================

    print_section(
        "STEP 6 — BUILDING HARMONY REQUEST"
    )


    print(
        "\nBuilding spatial + temporal subset..."
    )


    request = Request(

        collection=Collection(
            concept_id
        ),

        spatial=MUMBAI_BBOX,

        temporal={
            "start": START_TIME,
            "stop": END_TIME,
        },

        variables=[
            "precipitation",
        ],
    )


    print(
        "Harmony request created."
    )


    # ========================================================
    # STEP 7 — VALIDATE REQUEST
    # ========================================================

    print_section(
        "STEP 7 — VALIDATING HARMONY REQUEST"
    )


    try:

        is_valid = request.is_valid()

    except Exception as error:

        print(
            "\nCould not validate Harmony request."
        )

        print(
            f"Error: {error}"
        )

        raise


    if not is_valid:

        print(
            "\nHarmony request is INVALID."
        )


        print(
            "\nValidation messages:"
        )


        for message in (
            request.validation_messages
        ):

            print(
                f"- {message}"
            )


        raise RuntimeError(
            "Harmony request validation failed."
        )


    print(
        "Harmony request is VALID."
    )


    # ========================================================
    # STEP 8 — SUBMIT JOB
    # ========================================================

    print_section(
        "STEP 8 — SUBMITTING HARMONY JOB"
    )


    print(
        "\nSubmitting request..."
    )


    print(
        "Dataset:"
    )

    print(
        f"{SHORT_NAME}.{VERSION}"
    )


    print(
        "\nTime:"
    )

    print(
        f"{START_TIME} → {END_TIME}"
    )


    print(
        "\nArea:"
    )

    print(
        "Mumbai bounding box"
    )


    try:

        job_id = harmony_client.submit(
            request
        )

    except Exception as error:

        print(
            "\nHarmony submission failed."
        )

        print(
            f"Error: {error}"
        )

        raise


    print(
        "\nHarmony job submitted successfully!"
    )


    print(
        "Job ID:"
    )

    print(
        job_id
    )


    # ========================================================
    # STEP 9 — WAIT FOR PROCESSING
    # ========================================================

    print_section(
        "STEP 9 — WAITING FOR NASA PROCESSING"
    )


    print(
        "\nWaiting for NASA Harmony..."
    )


    try:

        harmony_client.wait_for_processing(
            job_id,
            show_progress=True,
        )

    except Exception as error:

        print(
            "\nHarmony processing failed."
        )

        print(
            f"Error: {error}"
        )

        raise


    print(
        "\nHarmony processing completed."
    )


    # ========================================================
    # STEP 10 — RESULT METADATA
    # ========================================================

    print_section(
        "STEP 10 — READING RESULT METADATA"
    )


    try:

        result = (
            harmony_client.result_json(
                job_id
            )
        )

    except Exception as error:

        print(
            "\nCould not retrieve Harmony result."
        )

        print(
            f"Error: {error}"
        )

        raise


    print(
        "\nHarmony result received."
    )


    if isinstance(
        result,
        dict,
    ):

        print(
            "\nReturned fields:"
        )


        for key in result.keys():

            print(
                f"- {key}"
            )


        if result.get(
            "request"
        ):

            print(
                "\nRequest URL:"
            )

            print(
                result[
                    "request"
                ]
            )


        if result.get(
            "outputDataSize"
        ):

            print(
                "\nOutput size:"
            )

            print(
                result[
                    "outputDataSize"
                ]
            )


    # ========================================================
    # STEP 11 — DOWNLOAD RESULT
    # ========================================================

    print_section(
        "STEP 11 — DOWNLOADING IMERG SUBSET"
    )


    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )


    try:

        futures = (
            harmony_client.download_all(
                job_id,
                directory=str(
                    OUTPUT_DIR
                ),
                overwrite=False,
            )
        )

    except Exception as error:

        print(
            "\nHarmony download failed."
        )

        print(
            f"Error: {error}"
        )

        raise


    downloaded_files = []


    for future in futures:

        try:

            file_path = (
                future.result()
            )

            downloaded_files.append(
                file_path
            )

        except Exception as error:

            print(
                "\nA downloaded file could not "
                "be retrieved."
            )

            print(
                f"Error: {error}"
            )

            raise


    # ========================================================
    # STEP 12 — FINAL SUMMARY
    # ========================================================

    print_section(
        "TEST COMPLETE"
    )


    print(
        "\n2010 IMERG Mumbai subset:"
    )

    print(
        "SUCCESS"
    )


    print(
        "\nDownloaded files:"
    )


    if downloaded_files:

        for file_path in downloaded_files:

            print(
                f"- {file_path}"
            )

    else:

        print(
            "No files were returned."
        )


    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )


    print(
        "\nNext step:"
    )

    print(
        "Use this same pipeline to build "
        "the complete 2010–2025 IMERG dataset."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()