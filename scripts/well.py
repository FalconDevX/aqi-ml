import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "PM1.csv"
OUTPUT = ROOT / "data" / "output.csv"

df = pd.read_csv(INPUT)

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%d.%m.%Y %H:%M"
)

df["Time"] = df["Time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

df.to_csv(OUTPUT, index=False)

print("Done!")