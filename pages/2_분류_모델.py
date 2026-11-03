# pages/2_분류_모델.py
# 뇌졸중 예측 분류 모델: 로지스틱 회귀와 의사결정트리

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
RANDOM_STATE = 42

st.set_page_config(page_title="분류 모델 | 뇌졸중 예측 실습실", page_icon="🌳", layout="wide")
st.title("🌳 분류 모델")
st.write(
    "뇌졸중(stroke)을 양성으로 두고 두 가지 분류 모델을 학습합니다. "
    "테스트 데이터는 모델 학습이나 크기 맞추기에 사용하지 않습니다."
)


@st.cache_data
def 데이터_읽기():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = 데이터_읽기()
except Exception as error:
    st.error("데이터를 읽지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()

필수열 = ["id", "stroke"] + list(FEATURE_LABELS.keys())
없는열 = [열 for 열 in 필수열 if 열 not in df.columns]
if 없는열:
    st.error("데이터에 필요한 열이 없습니다: " + ", ".join(없는열))
    st.stop()

df = df.sort_values("id", kind="stable").reset_index(drop=True)
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")
df = df.dropna(subset=["stroke"]).copy()
df["stroke"] = df["stroke"].astype(int)

st.subheader("1. 입력 속성 고르기")
선택된_우리말 = st.multiselect(
    "모델이 사용할 속성을 선택하세요.",
    options=list(FEATURE_LABELS.values()),
    default=[FEATURE_LABELS[c] for c in DEFAULT_FEATURES],
)
역변환 = {우리말: 열 for 열, 우리말 in FEATURE_LABELS.items()}
선택열 = [역변환[이름] for 이름 in 선택된_우리말]

if len(선택열) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()

if df["stroke"].nunique() < 2:
    st.error("stroke 열에 0과 1이 모두 있어야 분류 모델을 학습할 수 있습니다.")
    st.stop()

# ID 순으로 정렬한 뒤, 각 10명 묶음에서 앞 3명을 테스트용으로 고정한다.
행번호 = pd.Series(range(len(df)), index=df.index)
테스트마스크 = (행번호 % 10) < 3
훈련_df = df.loc[~테스트마스크].copy()
테스트_df = df.loc[테스트마스크].copy()

st.caption(
    f"전체 {len(df):,}명 · 훈련용 {len(훈련_df):,}명 · 테스트용 {len(테스트_df):,}명 "
    "(각 10명 중 앞 3명은 테스트용)"
)
if len(테스트_df) != 1533:
    st.warning(
        f"현재 데이터의 테스트용 인원은 {len(테스트_df):,}명입니다. "
        "요청한 1,533명과 다르다면 원본 데이터의 행 수를 확인해 주세요."
    )

X_train_raw = 훈련_df[선택열].apply(pd.to_numeric, errors="coerce")
X_test_raw = 테스트_df[선택열].apply(pd.to_numeric, errors="coerce")
y_train = 훈련_df["stroke"].astype(int)
y_test = 테스트_df["stroke"].astype(int)

# 결측값 채우기는 선택한 속성에만 적용하며, 중앙값은 훈련용 데이터에서만 계산한다.
