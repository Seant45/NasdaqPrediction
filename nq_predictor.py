import yfinance as yf
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("QQQ Movement Predictor")

@st.cache_data
def load_data():
    df = yf.download("QQQ", start="2023-01-01", interval="1d")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df["Return_1d"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_ratio"] = df["MA_5"] / df["MA_20"]
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

df = load_data()

st.subheader("Price Chart")
st.line_chart(df["Close"])

st.subheader("Latest Indicators")
latest = df.iloc[-1][["Return_1d", "MA_5", "MA_20", "MA_ratio"]]
st.write(latest.to_frame().T)

features = ["Return_1d", "MA_5", "MA_20", "MA_ratio"]
X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = (preds == y_test).mean()

st.subheader("Model Accuracy")
st.write(f"{accuracy:.2%}")

st.subheader("Last 5 Predictions")
table = pd.DataFrame({
    "Date": X_test.index[-5:],
    "Predicted": preds[-5:],
    "Actual": y_test[-5:]
})
table["Predicted"] = table["Predicted"].map({1: "Up", 0: "Down"})
table["Actual"] = table["Actual"].map({1: "Up", 0: "Down"})
st.table(table)

st.subheader("Tomorrow’s Prediction")

inputs = df[features].iloc[-1:].values
future_pred = model.predict(inputs)[0]
future_prob = model.predict_proba(inputs)[0]

move = "Up" if future_pred == 1 else "Down"
conf = max(future_prob)

st.write(f"Prediction: **{move}**")
st.write(f"Confidence: **{conf:.2%}**")

with st.expander("Show Indicators Used"):
    st.write(latest.to_frame().T)
