import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from features import create_features

BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE / "best_model.pkl")
    prep = joblib.load(BASE / "preprocessor.pkl")
    info = joblib.load(BASE / "split_info.pkl")
    return model, prep, info

model, prep, info = load_model()

st.title("🚕 NYC Taxi Trip Duration Prediction")
st.write("Ứng dụng AI dự đoán thời gian di chuyển của taxi tại New York")

# 1. Nhập thông tin chuyến xe
with st.form("taxi_form"):
    st.subheader("Thông tin chuyến đi")

    c1, c2 = st.columns(2)

    with c1:
        vendor = st.selectbox("Nhà vận tải", ["1", "2"])
        passengers = st.number_input("Số hành khách", 1, 8, 1)
        month = st.number_input("Tháng", 1, 12, 6)
        day = st.number_input("Thứ trong tuần (0-6)", 0, 6, 0)
        hour = st.number_input("Giờ đón", 0, 23, 8)
        minute = st.selectbox("Mốc phút", [0, 15, 30, 45])

    with c2:
        pickup_lat = st.number_input(
            "Vĩ độ điểm đón", value=40.7580, format="%.6f")
        pickup_lon = st.number_input(
            "Kinh độ điểm đón", value=-73.9855, format="%.6f")
        dropoff_lat = st.number_input(
            "Vĩ độ điểm trả", value=40.7308, format="%.6f")
        dropoff_lon = st.number_input(
            "Kinh độ điểm trả", value=-73.9973, format="%.6f")
        flag = st.selectbox("Store and Forward", ["N", "Y"])

    # Tính khoảng cách đường chim bay bằng Haversine
    R = 6371
    lat1, lat2 = np.radians([pickup_lat, dropoff_lat])
    dlat = lat2 - lat1
    dlon = np.radians(dropoff_lon - pickup_lon)
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    distance = 2 * R * np.arcsin(np.sqrt(a))

    st.write(f"Khoảng cách ước tính: {distance:.2f} km")
    submit = st.form_submit_button("Dự đoán thời gian")

# 2. Dự đoán
if submit:
    data = pd.DataFrame([{
        "vendor_id": vendor,
        "passenger_count": passengers,
        "pickup_month": month,
        "pickup_dayofweek": day,
        "pickup_hour": hour,
        "pickup_minute_bucket": minute,
        "pickup_lon_approx": pickup_lon,
        "pickup_lat_approx": pickup_lat,
        "dropoff_lon_approx": dropoff_lon,
        "dropoff_lat_approx": dropoff_lat,
        "straight_line_km_approx": distance,
        "store_and_fwd_flag": flag
    }])

    X = create_features(data)[info["columns"]]
    X = prep.transform(X)

    pred = max(float(np.expm1(model.predict(X)[0])), 1e-6)

    st.success("Dự đoán thành công!")
    st.metric("Thời gian dự đoán", f"{pred:.0f} giây")
    st.metric("Tương đương", f"{pred / 60:.2f} phút")
