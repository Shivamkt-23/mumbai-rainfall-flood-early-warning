from datetime import datetime, timezone, timedelta

from imerg_mumbai import (
    login_earthdata,
    get_imerg_mumbai
)


def get_hourly_imerg_mumbai(hour_start):
    """
    Calculate hourly rainfall from two IMERG
    30-minute products.

    Example:
        17:00 + 17:30
        = rainfall from 17:00 to 18:00 UTC
    """

    if hour_start.tzinfo is None:
        raise ValueError(
            "hour_start must be timezone-aware."
        )

    hour_start = hour_start.astimezone(
        timezone.utc
    )

    first_half = get_imerg_mumbai(
        hour_start
    )

    second_half = get_imerg_mumbai(
        hour_start + timedelta(minutes=30)
    )

    rainfall_1 = first_half[
        "rainfall_depth_mm"
    ]

    rainfall_2 = second_half[
        "rainfall_depth_mm"
    ]

    hourly_rainfall = (
        rainfall_1
        + rainfall_2
    )

    return {
        "timestamp": hour_start.isoformat(),
        "rainfall_0_30_mm": rainfall_1,
        "rainfall_30_60_mm": rainfall_2,
        "hourly_rainfall_mm": hourly_rainfall,
        "latitude": first_half["latitude"],
        "longitude": first_half["longitude"],
        "source": "NASA GPM IMERG Final V07"
    }


if __name__ == "__main__":

    print("Logging into NASA Earthdata...")

    login_earthdata()

    test_hour = datetime(
        2024,
        9,
        25,
        17,
        0,
        tzinfo=timezone.utc
    )

    print(
        "\nCalculating hourly IMERG rainfall..."
    )

    result = get_hourly_imerg_mumbai(
        test_hour
    )

    print()
    print("=" * 60)
    print("HOURLY IMERG MUMBAI RAINFALL")
    print("=" * 60)

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print("=" * 60)