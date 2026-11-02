# Data card: Online Retail II

- **Source:** Chen, D. (2019). *Online Retail II.* UCI Machine Learning Repository, id 502
  (SHA-256 of the zip `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`, checked
  by `scripts/prepare_data.py`). **Licence:** CC BY 4.0. Not committed.
- **Content:** 1,067,371 invoice lines from a UK online gift retailer, 2009-12-01 to
  2011-12-09. After dropping lines without a customer id: 824,364 lines, 44,876 invoices,
  5,942 customers. Invoices starting with "C" are cancellations or returns.
- **Derived task:** monthly snapshots (2010-03 to 2011-11). Customers with at least one prior
  purchase. Label = any purchase in the next 30 days (positive rate 21.5%, 84,578 rows).
  The split is temporal: train ≤ 2011-04, validation 2011-05/06, test 2011-07 to 2011-11.
- **Personal data:** customer ids are pseudonymous integers. No names or addresses. Country is
  present but unused.
- **Caveats:** one retailer with a heavily UK, wholesale-skewed customer base. Gift-retail
  seasonality (November peak) falls inside the test period.
