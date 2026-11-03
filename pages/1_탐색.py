# pages/1_탐색.py — 나이·혈당의 분포, 두 그룹 비교, 고혈압·심장병별 비율, 빈 값 살펴보기
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="탐색", page_icon="🔍", layout="wide")
st.title("🔍 탐색")
st.write("모델을 만들기 전에 데이터를 들여다봅니다. 화면의 값을 교재의 빈 표에 적으세요.")

# 사용자가 지정한 데이터 주소를 그대로 사용합니다.
데이터주소 = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def 데이터_읽기():
    return pd.read_csv(데이터주소, encoding="utf-8")


df = 데이터_읽기()
df["뇌졸중"] = df["stroke"].map({1: "뇌졸중 있음", 0: "뇌졸중 없음"})

st.subheader("1. 나이와 평균 혈당은 어떻게 퍼져 있는가")
왼쪽, 오른쪽 = st.columns(2)

with 왼쪽:
    나이_그래프 = px.histogram(
        df,
        x="age",
        nbins=40,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    st.plotly_chart(나이_그래프, width="stretch")

with 오른쪽:
    혈당_그래프 = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=40,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
    )
    st.plotly_chart(혈당_그래프, width="stretch")

st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람은 무엇이 다른가")
왼쪽, 오른쪽 = st.columns(2)

with 왼쪽:
    나이_상자그림 = px.box(
        df,
        x="뇌졸중",
        y="age",
        color="뇌졸중",
        title="나이 비교",
        labels={"age": "나이"},
    )
    st.plotly_chart(나이_상자그림, width="stretch")

with 오른쪽:
    혈당_상자그림 = px.box(
        df,
        x="뇌졸중",
        y="avg_glucose_level",
        color="뇌졸중",
        title="평균 혈당 비교",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    st.plotly_chart(혈당_상자그림, width="stretch")

두그룹 = (
    df.groupby("뇌졸중")
    .agg(
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
        고혈압_비율=("hypertension", "mean"),
        심장병_비율=("heart_disease", "mean"),
        사람_수=("id", "size"),
    )
    .reset_index()
)
두그룹["평균_나이"] = 두그룹["평균_나이"].round(1)
두그룹["평균_혈당"] = 두그룹["평균_혈당"].round(1)
두그룹["고혈압_비율"] = (두그룹["고혈압_비율"] * 100).round(1).astype(str) + "%"
두그룹["심장병_비율"] = (두그룹["심장병_비율"] * 100).round(1).astype(str) + "%"
st.dataframe(두그룹, width="stretch", hide_index=True)

st.subheader("3. 고혈압과 심장병이 있으면 뇌졸중 비율이 다른가")


def 비율표(열, 이름):
    """0과 1로 적힌 열마다 뇌졸중 비율을 구합니다."""
    표 = df.groupby(열)["stroke"].mean().reset_index()
    표[열] = 표[열].map({1: f"{이름} 있음", 0: f"{이름} 없음"})
    표["뇌졸중 비율(%)"] = (표["stroke"] * 100).round(2)
    return 표.rename(columns={열: "구분"})[["구분", "뇌졸중 비율(%)"]]
