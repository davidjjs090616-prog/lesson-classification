from pathlib import Path
import json

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
APP_DIR = Path(__file__).resolve().parent
RECORD_DIR = APP_DIR / "내_기록"
RECORD_FILE = RECORD_DIR / "stroke_data_notes.json"

EXPECTED_COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def load_notes():
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}
    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        return {
            "meanings": saved.get("meanings", {}),
            "source": saved.get("source", ""),
        }
    except (OSError, json.JSONDecodeError):
        st.warning("기록 파일을 읽지 못했습니다. 다시 입력해 저장해 주세요.")
        return {"meanings": {}, "source": ""}


def save_notes(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"meanings": meanings, "source": source}
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def describe_value_kind(series):
    if pd.api.types.is_bool_dtype(series.dtype):
        return "논리형"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "정수형"
    if pd.api.types.is_float_dtype(series.dtype):
        return "실수형"
    return "문자형"


st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고, 열의 의미와 출처를 교재를 참고해 직접 기록해 보세요.")

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

saved_notes = load_notes()
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_rate = stroke_count / total_people * 100 if total_people else 0.0

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_people:,}")
card2.metric("열 개수", f"{column_count:,}")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}")
card4.metric("stroke가 1인 비율", f"{stroke_rate:.2f}%")

st.divider()
st.header("1. 열 정보")
st.write("우리말 뜻은 교재를 참고해 직접 입력한 뒤 저장하세요.")

info_rows = [
    {
        "열 이름": column,
        "우리말 뜻": saved_notes["meanings"].get(column, ""),
        "값의 종류": describe_value_kind(df[column]),
        "빈 값 개수": int(df[column].isna().sum()),
    }
    for column in df.columns
]
info_df = pd.DataFrame(info_rows)

with st.form("notes_form"):
    edited_info = st.data_editor(
        info_df,
        hide_index=True,
        use_container_width=True,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름", width="medium"),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻",
                help="교재를 참고해 열의 뜻을 입력하세요.",
                width="large",
            ),
            "값의 종류": st.column_config.TextColumn("값의 종류", width="small"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", width="small"),
        },
        key="column_info_editor",
    )
    st.header("2. 데이터 출처")
    source_text = st.text_area(
        "교재에 나온 데이터 출처를 입력하세요.",
        value=saved_notes["source"],
        placeholder="교재에 적힌 데이터 출처를 입력하세요.",
        height=100,
        key="source_editor",
    )
    save_clicked = st.form_submit_button("우리말 뜻과 출처 저장", type="primary")

if save_clicked:
    meanings = {}
    for _, row in edited_info.iterrows():
        value = row["우리말 뜻"]
        meanings[str(row["열 이름"])] = "" if pd.isna(value) else str(value)
    save_notes(meanings, source_text)
    st.success(f"입력한 내용을 저장했습니다: 내_기록/{RECORD_FILE.name}")
    st.rerun()

st.divider()
st.header("3. 데이터 처음 다섯 줄")
st.write("CSV에서 읽은 원본 데이터의 처음 다섯 줄입니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

st.divider()
st.caption(f"데이터 출처 URL: {DATA_URL}")
