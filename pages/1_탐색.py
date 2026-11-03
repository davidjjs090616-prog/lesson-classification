import pandas as pd
import plotly.express as px
import streamlit as st


DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
EXPECTED_COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def stroke_rate_percent(series):
    if len(series) == 0:
        return 0.0
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float((numeric == 1).sum() / len(numeric) * 100)


def make_binary_rate_table(data, column):
    work = data[[column, "stroke"]].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work["stroke"] = pd.to_numeric(work["stroke"], errors="coerce")
    work = work.dropna(subset=[column, "stroke"])
    work = work[work[column].isin([0, 1])]

    summary = (
        work.groupby(column)["stroke"]
        .agg(["size", "mean"])
        .reset_index()
        .rename(
            columns={
                column: "구분",
                "size": "사람 수",
                "mean": "뇌졸중 비율(%)",
            }
        )
    )
    summary["구분"] = summary["구분"].map({0: "없음 (0)", 1: "있음 (1)"})
    summary["뇌졸중 비율(%)"] = summary["뇌졸중 비율(%)"] * 100
    return summary


st.title("📊 데이터 탐색")
st.caption("뇌졸중 데이터의 분포와 그룹별 차이를 그래프와 표로 살펴봅니다.")

try:
    df = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.caption(f"오류 정보: {exc}")
    st.stop()

missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
if missing_columns:
    st.error("예상한 열을 데이터에서 찾지 못했습니다: " + ", ".join(missing_columns))
    st.stop()

plot_df = df.copy()
numeric_columns = [
    "age", "avg_glucose_level", "hypertension", "heart_disease", "stroke"
]
for column in numeric_columns:
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

plot_df["뇌졸중 여부"] = plot_df["stroke"].map(
    {0: "뇌졸중 없음", 1: "뇌졸중 있음"}
)

st.header("1. 나이와 평균 혈당 분포")
left, right = st.columns(2)

with left:
    age_fig = px.histogram(
        plot_df.dropna(subset=["age"]),
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    age_fig.update_layout(xaxis_title="나이", yaxis_title="사람 수")
    st.plotly_chart(age_fig, use_container_width=True)
