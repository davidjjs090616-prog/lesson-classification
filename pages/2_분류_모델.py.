# pages/2_분류_모델.py
# 뇌졸중 예측 분류 모델: 로지스틱 회귀와 의사결정트리

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
LABEL_TO_COLUMN = {label: column for column, label in FEATURE_LABELS.items()}
DEFAULT_FEATURES = ["나이", "평균 혈당", "고혈압", "심장병"]
REQUIRED_COLUMNS = ["id", "stroke"] + list(FEATURE_LABELS.keys())
RANDOM_STATE = 42


st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧪",
    layout="wide",
)


@st.cache_data
def 데이터_읽기():
    """첫 화면과 탐색 페이지에서 사용하는 CSV를 UTF-8로 읽습니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


def 정확도_문자열(값):
    return f"{값 * 100:.2f}%"


def dot_문자열_이스케이프(문자열):
    """DOT의 따옴표 문자열 안에서 특수문자를 안전하게 표현합니다."""
    return (
        str(문자열)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def 결정트리_dot_만들기(모델, 학습_자료, 정답, 선택_열, 크기맞춤기):
    """훈련 데이터의 각 노드 표본 수와 양성 수를 표시하는 DOT 문자열을 만듭니다."""
    나무 = 모델.tree_
    왼쪽자식 = 나무.children_left
    오른쪽자식 = 나무.children_right
    질문한_열 = 나무.feature
    질문값 = 나무.threshold

    # 각 훈련용 행이 어느 노드까지 도달했는지 기록합니다.
    도달경로 = 모델.decision_path(학습_자료)

    dot = [
        "digraph DecisionTree {",
        'graph [rankdir=TB, bgcolor="transparent", nodesep="0.35", ranksep="0.55"];',
        'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10, color="#64748b", margin="0.12,0.08"];',
        'edge [fontname="sans-serif", fontsize=10, color="#475569", arrowsize=0.7];',
    ]

    잎_번호 = []
    잎_아님_번호 = []
    실제로_물은_열 = set()

    for 노드 in range(나무.node_count):
        마스크 = 도달경로[:, 노드].toarray().ravel().astype(bool)
        그마디_사람수 = int(마스크.sum())
        그마디_뇌졸중수 = int(정답.loc[마스크].sum())
        뇌졸중비율 = (
            그마디_뇌졸중수 / 그마디_사람수 * 100
            if 그마디_사람수
            else 0.0
        )

        잎인지 = 왼쪽자식[노드] == 오른쪽자식[노드]

        if 잎인지:
            잎_번호.append(노드)
            클래스별_값 = list(나무.value[노드][0])
            가장큰_클래스_위치 = max(
                range(len(클래스별_값)),
                key=lambda 위치: 클래스별_값[위치],
            )
            예측_클래스 = int(모델.classes_[가장큰_클래스_위치])
