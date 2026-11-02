"""Online Retail II (UCI, CC BY 4.0) -> invoice-level events (data/invoices.parquet).

One event per (customer, invoice): time, revenue, number of lines, and whether it is a return
(cancelled invoices start with 'C'). Rows without a customer id are dropped.
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"

d = Path("data")
d.mkdir(exist_ok=True)
z = d / "online_retail_ii.zip"
if not z.exists():
    urllib.request.urlretrieve(URL, z)
if hashlib.sha256(z.read_bytes()).hexdigest() != SHA256:
    raise SystemExit("checksum mismatch")
with zipfile.ZipFile(z) as zf:
    zf.extract("online_retail_II.xlsx", d)
sheets = pd.read_excel(d / "online_retail_II.xlsx", sheet_name=None,
                       dtype={"Invoice": str, "StockCode": str})
df = pd.concat(sheets.values(), ignore_index=True)
df = df.dropna(subset=["Customer ID"])
df["customer"] = df["Customer ID"].astype("int64")
df["revenue"] = df["Quantity"] * df["Price"]
df["is_return"] = df["Invoice"].str.startswith("C")
inv = (df.groupby(["customer", "Invoice"], as_index=False)
         .agg(time=("InvoiceDate", "min"), revenue=("revenue", "sum"),
              lines=("StockCode", "size"), is_return=("is_return", "first")))
inv["t"] = (inv["time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)  # unit-safe
inv.to_parquet(d / "invoices.parquet", index=False)
print(len(df), "lines ->", len(inv), "invoices,", inv["customer"].nunique(), "customers,",
      inv["time"].min(), "->", inv["time"].max())
