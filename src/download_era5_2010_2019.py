from pathlib import Path
import cdsapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ERA5_2010_2019"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


client = cdsapi.Client()

dataset = "reanalysis-era5-single-levels-timeseries"

request = {
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "total_precipitation"
    ],
    "location": {
        "longitude": 73,
        "latitude": 19
    },
    "date": [
        "2010-01-01/2019-12-31"
    ],
    "data_format": "csv"
}

target = (
    OUTPUT_DIR
    / "era5_2010_2019_raw.csv"
)

print("=" * 60)
print("ERA5 DOWNLOAD")
print("=" * 60)
print("Period: 2010-01-01 → 2019-12-31")
print("Location: 19°N, 73°E")
print("Output:", target)
print("=" * 60)

client.retrieve(
    dataset,
    request,
    str(target)
)

print("\nDOWNLOAD COMPLETE")
print("Saved to:", target)