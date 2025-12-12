# app.py (Streamlit)
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Put your CSV in the same folder:
#   absher_suspicious_dataset_ready.csv

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


CSV_DEFAULT = "absher_suspicious_dataset_ready.csv"


def mode_or_first(s: pd.Series):
    m = s.mode()
    return m.iloc[0] if len(m) else s.iloc[0]


def build_baseline(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("user_id")
        .agg(
            {
                "device_type": mode_or_first,
                "device_status": mode_or_first,
                "location_status": mode_or_first,
                "time_bucket": mode_or_first,
                "network_type": mode_or_first,
                "failed_attempts": "mean",
            }
        )
        .rename(
            columns={
                "device_type": "base_device_type",
                "device_status": "base_device_status",
                "location_status": "base_location_status",
                "time_bucket": "base_time_bucket",
                "network_type": "base_network_type",
                "failed_attempts": "base_failed_attempts_mean",
            }
        )
        .reset_index()
    )


def add_behavior_features(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    df2 = df.merge(baseline, on="user_id", how="left")

    df2["dev_device_type"] = (df2["device_type"] != df2["base_device_type"]).astype(int)
    df2["dev_device_status"] = (df2["device_status"] != df2["base_device_status"]).astype(int)
    df2["dev_location"] = (df2["location_status"] != df2["base_location_status"]).astype(int)
    df2["dev_time"] = (df2["time_bucket"] != df2["base_time_bucket"]).astype(int)
    df2["dev_network"] = (df2["network_type"] != df2["base_network_type"]).astype(int)
    df2["dev_failed_high"] = (
        df2["failed_attempts"] > (df2["base_failed_attempts_mean"] + 1)
    ).astype(int)

    return df2


def train_model(df2: pd.DataFrame):
    y = df2["is_suspicious"]

    feature_cols = [
        "request_type",
        "device_type",
        "device_status",
        "location_status",
        "time_bucket",
        "network_type",
        "vpn_used",
        "failed_attempts",
        "ip_reputation",
        "session_velocity",
        "impossible_travel",
        "previous_watchlist",
        "sensitive_action",
        "dev_device_type",
        "dev_device_status",
        "dev_location",
        "dev_time",
        "dev_network",
        "dev_failed_high",
    ]
    X = df2[feature_cols]

    cat_cols = [
        "request_type",
        "device_type",
        "device_status",
        "location_status",
        "time_bucket",
        "network_type",
        "ip_reputation",
        "session_velocity",
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test))
    return model, X.columns.tolist(), report


def predict_event(model, feature_cols, baseline_map, event: dict) -> int:
    b = baseline_map.get(
        event["user_id"],
        {
            "base_device_type": "mobile",
            "base_device_status": "known_device",
            "base_location_status": "normal_location",
            "base_time_bucket": "morning",
"base_network_type": "mobile_data",
            "base_failed_attempts_mean": 0,
        },
    )

    e = event.copy()
    e["dev_device_type"] = int(e["device_type"] != b["base_device_type"])
    e["dev_device_status"] = int(e["device_status"] != b["base_device_status"])
    e["dev_location"] = int(e["location_status"] != b["base_location_status"])
    e["dev_time"] = int(e["time_bucket"] != b["base_time_bucket"])
    e["dev_network"] = int(e["network_type"] != b["base_network_type"])
    e["dev_failed_high"] = int(e["failed_attempts"] > b["base_failed_attempts_mean"] + 1)

    X_one = pd.DataFrame([e])[feature_cols]
    return int(model.predict(X_one)[0])


# ---------------- UI ----------------

st.set_page_config(page_title="Absher Risk Engine Demo", layout="wide")
st.title("Absher Risk Engine (Demo)")
st.caption("Risk-based classification: Normal vs Suspicious → OTP vs Nafath")

with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    csv_path = st.text_input("Or use local CSV filename", value=CSV_DEFAULT)

@st.cache_data
def load_df(uploaded_file, csv_path: str):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(csv_path)

@st.cache_resource
def build_all(df: pd.DataFrame):
    baseline = build_baseline(df)
    df2 = add_behavior_features(df, baseline)
    model, feature_cols, report = train_model(df2)
    baseline_map = baseline.set_index("user_id").to_dict(orient="index")
    return baseline, df2, model, feature_cols, report, baseline_map

df = load_df(uploaded, csv_path)

baseline, df2, model, feature_cols, report, baseline_map = build_all(df)

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)

with colB:
    st.subheader("Model Report (Holdout Test)")
    st.code(report)

st.divider()
st.subheader("Live Event Simulation (Stream)")

# Input form
left, right = st.columns([1, 1])

with left:
    user_id = st.number_input("user_id", min_value=1, max_value=10_000_000, value=10, step=1)

    request_type = st.selectbox(
        "request_type",
        sorted(df["request_type"].unique().tolist())
    )
    device_type = st.selectbox(
        "device_type",
        sorted(df["device_type"].unique().tolist())
    )
    device_status = st.selectbox(
        "device_status",
        sorted(df["device_status"].unique().tolist())
    )
    location_status = st.selectbox(
        "location_status",
        sorted(df["location_status"].unique().tolist())
    )
    time_bucket = st.selectbox(
        "time_bucket",
        sorted(df["time_bucket"].unique().tolist())
    )
    network_type = st.selectbox(
        "network_type",
        sorted(df["network_type"].unique().tolist())
    )
    vpn_used = st.selectbox("vpn_used", [0, 1], index=0)
    failed_attempts = st.number_input("failed_attempts", min_value=0, max_value=10, value=0, step=1)

with right:
    ip_reputation = st.selectbox(
        "ip_reputation",
        sorted(df["ip_reputation"].unique().tolist())
    )
    session_velocity = st.selectbox(
        "session_velocity",
        sorted(df["session_velocity"].unique().tolist())
    )
    impossible_travel = st.selectbox("impossible_travel", [0, 1], index=0)
    previous_watchlist = st.selectbox("previous_watchlist", [0, 1], index=0)
    sensitive_action = st.selectbox("sensitive_action", [0, 1], index=0)

event = {
    "user_id": int(user_id),
    "request_type": request_type,
    "device_type": device_type,
    "device_status": device_status,
    "location_status": location_status,
    "time_bucket": time_bucket,
    "network_type": network_type,
    "vpn_used": int(vpn_used),
    "failed_attempts": int(failed_attempts),
    "ip_reputation": ip_reputation,
    "session_velocity": session_velocity,
    "impossible_travel": int(impossible_travel),
    "previous_watchlist": int(previous_watchlist),
    "sensitive_action": int(sensitive_action),
}

if st.button("Analyze Event"):pred = predict_event(model, feature_cols, baseline_map, event)
    verdict = "Suspicious" if pred == 1 else "Normal"
    action = "Require Nafath (strong auth)" if pred == 1 else "OTP / standard auth"

    st.success(f"Prediction: **{verdict}**")
    st.info(f"Decision: **{action}**")

    # Show behavior deviation explanation
    b = baseline_map.get(event["user_id"])
    if b:
        st.subheader("User Baseline (for comparison)")
        st.json(b)
    else:
        st.warning("No baseline found for this user_id (treated as new user).")


st.divider()
st.subheader("Quick Demo Buttons")

c1, c2 = st.columns(2)

with c1:
    if st.button("Try Normal Example"):
        sample = {
            "user_id": 10,
            "request_type": "login",
            "device_type": "mobile",
            "device_status": "known_device",
            "location_status": "normal_location",
            "time_bucket": "morning",
            "network_type": "mobile_data",
            "vpn_used": 0,
            "failed_attempts": 0,
            "ip_reputation": "good",
            "session_velocity": "normal",
            "impossible_travel": 0,
            "previous_watchlist": 0,
            "sensitive_action": 0,
        }
        pred = predict_event(model, feature_cols, baseline_map, sample)
        st.write(sample)
        st.success("Normal" if pred == 0 else "Suspicious")

with c2:
    if st.button("Try Suspicious Example"):
        sample = {
            "user_id": 10,
            "request_type": "change_phone",
            "device_type": "desktop",
            "device_status": "new_device",
            "location_status": "foreign_country",
            "time_bucket": "night",
            "network_type": "vpn",
            "vpn_used": 1,
            "failed_attempts": 4,
            "ip_reputation": "bad",
            "session_velocity": "very_fast",
            "impossible_travel": 1,
            "previous_watchlist": 1,
            "sensitive_action": 1,
        }
        pred = predict_event(model, feature_cols, baseline_map, sample)
        st.write(sample)
        st.success("Normal" if pred == 0 else "Suspicious")
