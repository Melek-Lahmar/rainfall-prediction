# Data sources and preparation

The repository does not redistribute the source CSV files. The Australian course copy has unclear redistribution terms, while the Tunisia and Italy files are generated on demand from Open-Meteo. This page records the provenance, schema, and preparation rules needed to reproduce the experiments.

## Australia

- Local filename: `weatherAUS_2.csv`
- Source described by the IBM course notebook: Australian Bureau of Meteorology observations distributed through Kaggle's [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/) dataset
- Raw period: 1 November 2007 to 25 June 2017
- Raw shape: 145,460 rows and 23 columns
- Scope used here: Melbourne, Melbourne Airport, and Watsonia
- Modeling sample: 7,557 complete rows after dropping all rows with at least one missing value and applying the location filter

Place `weatherAUS_2.csv` in the repository root before running the Australian notebook or build script. The file is intentionally ignored by Git.

The course workflow predicts the original `RainTomorrow` label from the weather measurements on the same row. For readability, the extension renames original `RainToday` to `RainYesterday` and original `RainTomorrow` to the modeling target `RainToday`. This is a naming convention only; the task remains next-day rainfall classification.

## Tunisia and Italy

The multi-country notebook requests historical ERA5 reanalysis through the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api):

| Experiment | Location | Requested period | Evaluation period |
|---|---|---|---|
| Tunisia | Sfax | 2015-01-01 to 2025-12-31 | 2024-01-01 to 2025-12-30 |
| Italy | Rome | 2015-01-01 to 2025-12-31 | 2024-01-01 to 2025-12-30 |

The final requested day has no following-day label, so each generated city table contains 4,017 labeled rows ending on 30 December 2025. The supplied execution screenshots report no missing values in either generated table.

The notebook creates these local outputs:

- `Tunisie_weather_23_columns.csv`
- `Italie_weather_23_columns.csv`
- `METHODOLOGIE_ET_SOURCES.json`
- `Tunisie_Italie_ML_resultats.csv`
- `Tunisie_Italie_2_Datasets_Meteo.zip`

These generated files and the temporary API cache are ignored by Git. The notebook can export and later restore `weather_cache_partial.zip` if Open-Meteo returns HTTP 429 or a Colab session ends.

## Shared 23-column schema

The generated Sfax and Rome CSVs use the Australian column names and order:

```text
Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am,
WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow
```

Matching names do not mean identical physical measurements. In the Open-Meteo-derived files:

- `Evaporation` is FAO reference evapotranspiration (ET0), not measured evaporation.
- `Sunshine` is modeled daily sunshine duration converted from seconds to hours.
- `WindGustDir` is the 10 m wind direction at the hour containing the largest hourly gust, not a directly verified gust-direction observation.
- `Cloud9am` and `Cloud3pm` convert modeled cloud cover percentages to approximate octas.
- Temperatures, humidity, pressure, wind, and cloud values at 09:00 and 15:00 use each city's requested local time zone.

ERA5 is a gridded reanalysis combining observations and numerical modeling; it is not equivalent to a dedicated station record.

## Rain labels and chronological split

For Sfax and Rome:

- `RainToday = Yes` when daily precipitation is strictly greater than 1.0 mm.
- `RainTomorrow` is the next calendar day's `RainToday` value within the same city.
- The last date for each city is removed because its next-day label is unavailable.
- Training rows end on 30 December 2023.
- The test set begins on 1 January 2024.
- `Date` and `RainTomorrow` are excluded from the predictors.
- Numeric and categorical imputation and encoding are fitted inside a scikit-learn pipeline using training data only.

The 31 December 2023 feature row is excluded because its label describes rainfall on 1 January 2024, which belongs to the test period.

## Attribution and terms

- Australian course context: IBM *Machine Learning with Python* on Coursera; IBM does not endorse this repository.
- Australian data references: [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) and [Kaggle Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/).
- Tunisia and Italy: Weather data by [Open-Meteo.com](https://open-meteo.com/) under its stated attribution terms, using [ERA5 reanalysis](https://climate.copernicus.eu/climate-reanalysis).

Review the current source terms before redistributing any downloaded dataset.
