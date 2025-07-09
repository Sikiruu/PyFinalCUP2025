# Home Appliance Power Usage Dataset

This dataset contains timestamped power usage data for household appliances, specifically focusing on the **dishwasher**, **washing machine**, and overall **aggregate** power consumption. The filename indicates the household ID to which the data belongs.

---

## 📄 Dataset Overview

| Column Name      | Description                                                             |
|------------------|-------------------------------------------------------------------------|
| `Time`           | Timestamp of the measurement (format: `YYYY-MM-DD HH:MM:SS`)            |
| `Aggregate`      | Total household power usage at that timestamp (in watts)                |
| `dishwasher`     | Power consumption of the dishwasher at that timestamp (in watts)        |
| `washingmachine` | Power consumption of the washing machine at that timestamp (in watts)   |

---

## 📊 Exmaple

```
               Time           Aggregate  dishwasher  washingmachine
0   2013-11-28 12:15:35         267         0              0
1   2013-11-28 12:15:51         265         0              0
2   2013-11-28 12:16:06         267         0              0
3   2013-11-28 12:16:20         264         0              0
4   2013-11-28 12:16:21         266         0              0
```
---

## 📅 Date Range
two month
---


## 🛠 Usage Example

### Load the Data with Pandas

```python
import pandas as pd

df = pd.read_csv('your_dataset.csv', parse_dates=['Time'])
```

### Identify Active Appliance Periods

You can set a threshold (e.g., 10 watts) to detect when an appliance is likely active:

```python
df['washingmachine_active'] = df['washingmachine'] > 10
df['dishwasher_active'] = df['dishwasher'] > 10
```

---

## ⚠️ Notes

- **Sampling Interval:** Time intervals between rows are not fixed; resampling may be necessary for certain analyses.
- **Power Unit:** All consumption values are in **watts**.
- **Missing or Anomalous Data:** May need to be cleaned or interpolated.

---

