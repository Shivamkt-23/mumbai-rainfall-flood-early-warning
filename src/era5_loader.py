import sys
import numpy as np
import pandas as pd
import xarray as xr

def find_var(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None

def regional_series(ds, var):
    if var is None:
        return None
    da = ds[var]
    dims = [d for d in da.dims if d.lower() in ("latitude", "longitude", "lat", "lon")]
    if dims:
        da = da.mean(dim=dims, skipna=True)
    return np.asarray(da.values).reshape(-1)

def convert_era5(nc_path, output_csv):
    ds = xr.open_dataset(nc_path)

    tp = find_var(ds, ["tp", "total_precipitation"])
    t2m = find_var(ds, ["t2m", "2m_temperature"])
    d2m = find_var(ds, ["d2m", "2m_dewpoint_temperature"])
    msl = find_var(ds, ["msl", "mean_sea_level_pressure"])
    u10 = find_var(ds, ["u10", "10m_u_component_of_wind"])
    v10 = find_var(ds, ["v10", "10m_v_component_of_wind"])

    if tp is None:
        raise ValueError("Total precipitation variable not found (usually 'tp').")

    rain = regional_series(ds, tp)
    units = str(ds[tp].attrs.get("units", "")).lower()
    if units in ("m", "meter", "meters", "metre", "metres", ""):
        rain = rain * 1000.0

    out = pd.DataFrame({
        "datetime": pd.to_datetime(ds["time"].values),
        "rainfall_mm": rain
    })

    t = regional_series(ds, t2m)
    d = regional_series(ds, d2m)
    p = regional_series(ds, msl)
    u = regional_series(ds, u10)
    v = regional_series(ds, v10)

    out["temperature_c"] = (t - 273.15) if t is not None else np.nan
    if d is not None and t is not None:
        # Magnus approximation for relative humidity
        a, b = 17.625, 243.04
        tc = out["temperature_c"].to_numpy()
        dc = d - 273.15
        gamma_t = a * tc / (b + tc)
        gamma_d = a * dc / (b + dc)
        out["humidity_pct"] = 100 * np.exp(gamma_d - gamma_t)
    else:
        out["humidity_pct"] = np.nan

    out["pressure_hpa"] = (p / 100.0) if p is not None else np.nan
    if u is not None and v is not None:
        out["wind_speed_ms"] = np.sqrt(u*u + v*v)
    else:
        out["wind_speed_ms"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["datetime", "rainfall_mm"]).sort_values("datetime")
    out.to_csv(output_csv, index=False)
    print(f"Saved {len(out):,} rows to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/era5_loader.py input.nc output.csv")
        raise SystemExit(1)
    convert_era5(sys.argv[1], sys.argv[2])
