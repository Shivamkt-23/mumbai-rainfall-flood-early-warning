# Mumbai AI Rainfall & Flood-Risk Prototype

Beginner-friendly SIH college prototype for:
AI/ML-based heavy rainfall early warning and preliminary inundation-risk prediction.

## Current MVP
- Historical/demo weather data
- Rainfall history features
- Random Forest next-hour rainfall prediction
- Preliminary flood-risk score
- Interactive Mumbai map
- Historical-event replay

## Important
The included CSV is synthetic demo data so the application runs immediately.
For the real project, replace it with your ERA5 NetCDF data using `src/era5_loader.py`.

This is a college prototype, not an operational IMD warning system.

## Run
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Convert ERA5 NetCDF to CSV
```bash
python src/era5_loader.py data/your_era5_file.nc data/era5_clean.csv
```
