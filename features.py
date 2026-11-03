import pandas as pd
import numpy as np

# Hàm xử lý bất thường và tạo đặc trưng
def create_features(df):
    df = df.drop(columns=[
        "trip_id", "trip_duration", "dropoff_datetime"
    ], errors="ignore").copy()

    cat = ["vendor_id", "store_and_fwd_flag"]
    for c in df.columns.difference(cat):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.loc[df["passenger_count"] <= 0, "passenger_count"] = np.nan
    df.loc[~df["pickup_month"].between(1, 12), "pickup_month"] = np.nan
    df.loc[~df["pickup_dayofweek"].between(0, 6), "pickup_dayofweek"] = np.nan
    df.loc[~df["pickup_hour"].between(0, 23), "pickup_hour"] = np.nan
    df.loc[~df["pickup_minute_bucket"].isin([0, 15, 30, 45]),
           "pickup_minute_bucket"] = np.nan
    df.loc[df["straight_line_km_approx"] < 0,
           "straight_line_km_approx"] = np.nan

    for c in ["pickup_lat_approx", "dropoff_lat_approx"]:
        df.loc[~df[c].between(-90, 90), c] = np.nan

    for c in ["pickup_lon_approx", "dropoff_lon_approx"]:
        df.loc[~df[c].between(-180, 180), c] = np.nan

    # Đặc trưng bổ sung
    df["lat_diff"] = abs(df["dropoff_lat_approx"]
                         - df["pickup_lat_approx"])
    df["lon_diff"] = abs(df["dropoff_lon_approx"]
                         - df["pickup_lon_approx"])

    df["rush_hour"] = np.where(
        df["pickup_hour"].isna(), np.nan,
        (df["pickup_hour"].between(7, 9) |
         df["pickup_hour"].between(16, 19)).astype(int)
    )

    # Mã hóa phân loại được học trong Pipeline
    df["vendor_id"] = df["vendor_id"].astype("string")
    df["store_and_fwd_flag"] = (
        df["store_and_fwd_flag"].astype("string")
        .str.strip().str.upper()
    )
    df.loc[~df["store_and_fwd_flag"].isin(["Y", "N"]).fillna(False),
           "store_and_fwd_flag"] = pd.NA

    for c in cat:
        df[c] = df[c].fillna("Missing").astype(str)

    return df