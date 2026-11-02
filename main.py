from pathlib import Path
import json

import pandas as pd
import streamlit as st

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고, 각 열의 의미를 직접 정리해 보는 실습 공간입니다.")

# -----------------------------
# 데이터 및 기록 파일
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/davidjjs090616-prog/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "뇌졸중_예측_실습실.json"

COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi",
    "smoking_status", "stroke",
]

def load_record():
    """저장된 우리말 뜻과 데이터 출처를 불러옵니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}
    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return {"meanings": {}, "source": ""}
        return {
            "meanings": data.get("meanings", {}),
            "source": data.get("source", ""),
        }
    except (OSError, json.JSONDecodeError):
        return {"meanings": {}, "source": ""}

def save_record(meanings, source):
    """입력한 내용을 JSON UTF-8 파일로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {"meanings": meanings, "source": source},
            file,
            ensure_ascii=False,
            indent=2,
        )

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    # 요구한 열 순서를 유지하되, 원본에 열이 추가/누락된 경우에도 앱이 멈추지 않도록 처리
    present = [column for column in COLUMNS if column in df.columns]
    remaining = [column for column in df.columns if column not in present]
    return df[present + remaining]

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()

record = load_record()
meanings = record.get("meanings", {})
source_text = record.get("source", "")

# -----------------------------
# 첫 화면: 데이터 소개
# -----------------------------
st.header("1. 데이터 한눈에 보기")
st.write(
    "이 데이터는 사람들의 기본 정보와 건강 관련 특성, 그리고 뇌졸중(stroke) 발생 여부를 "
    "담고 있습니다. 먼저 데이터의 크기와 각 열의 구성을 확인해 보세요."
)

total_people = len(df)
column_count = len(df.columns)
stroke_count = int((pd.to_numeric(df["stroke"], errors="coerce") == 1).sum()) if "stroke" in df.columns else 0
stroke_rate = (stroke_count / total_people * 100) if total_people else 0

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{column_count:,}개")
