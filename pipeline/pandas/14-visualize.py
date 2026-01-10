#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to-2019-01-09.csv', ',')

# Remove Weighted_Price column
if "Weighted_Price" in df.columns:
    df = df.drop(columns=["Weighted_Price"])

# Rename Timestamp to Date
df = df.rename(columns={"Timestamp": "Date"})

# Convert timestamp values to datetime
df["Date"] = pd.to_datetime(df["Date"], unit="s")

# Index the DataFrame on Date
df = df.set_index("Date")

# Fill missing values
df["Close"] = df["Close"].fillna(method="ffill")
for col in ["High", "Low", "Open"]:
    df[col] = df[col].fillna(df["Close"])
for col in ["Volume_(BTC)", "Volume_(Currency)"]:
    df[col] = df[col].fillna(0)

# Filter data from 2017 onwards
df = df["2017-01-01":]

# Resample daily and aggregate
df = df.resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum"
})

# Return transformed DataFrame before plotting
print(df)

# Plot Close price
df["Close"].plot(title="Daily Bitcoin Close Price (Coinbase, 2017+)")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.show()
