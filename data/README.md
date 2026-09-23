# Dataset Notes

## Dataset

- Local filename: `weatherAUS_2.csv`
- Raw shape: 145,460 rows × 23 columns
- Observation period in the local file: 2007-11-01 through 2017-06-25
- Rows represent daily weather observations at Australian weather stations.
- Modeling target: original `RainTomorrow`, renamed `RainToday` under the IBM course's prediction-time convention.

## Source

The original IBM notebook attributes the observations to the [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) and states that the course dataset was obtained from Kaggle's [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/) package.

## Modeling Scope

The course workflow drops rows containing missing values and retains Melbourne, Melbourne Airport, and Watsonia. This leaves 7,557 observations.

## Local Setup

Place `weatherAUS_2.csv` in the repository root. The notebook loads it with:

```python
pd.read_csv("weatherAUS_2.csv")
```

The CSV is ignored by Git because the redistribution/license terms for the packaged course copy were not verified. Obtain the dataset through the course materials or the source links above and review their current terms before redistributing it.
