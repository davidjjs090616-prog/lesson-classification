import pandas as pd
import plotly.express as px
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 데이터 탐색", page_icon="📊", layout="wide"
)

st.title("📊 뇌졸중 데이터 탐색 실습실")
st.markdown(
    "데이터의 분포를 시각적으로 확인하고, 다양한 조건에 따른 뇌졸중 발생 경향을"
    " 분석해봅니다."
)
st.divider()


# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
  url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
  df = pd.read_csv(url, encoding="utf-8")
  return df


try:
  df = load_data()
except Exception as e:
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
  st.stop()


# 1. 나이와 평균 혈당의 분포 히스토그램
st.subheader("1. 나이 및 평균 혈당 분포")
col1, col2 = st.columns(2)

with col1:
  fig_age = px.histogram(
      df, x="age", title="나이(age) 분포", nbins=30, color_discrete_sequence=["#636EFA"]
  )
  fig_age.update_xaxis(title="나이")
  fig_age.update_yaxis(title="사람 수")
  st.plotly_chart(fig_age, use_container_width=True)

with col2:
  fig_glucose = px.histogram(
      df,
      x="avg_glucose_level",
      title="평균 혈당(avg_glucose_level) 분포",
      nbins=30,
      color_discrete_sequence=["#EF553B"],
  )
  fig_glucose.update_xaxis(title="평균 혈당")
  fig_glucose.update_yaxis(title="사람 수")
  st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()


# 2. 뇌졸중 여부에 따른 나이/평균 혈당 상자그림 및 평균값 표
st.subheader("2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")

df_plot = df.copy()
df_plot["stroke_label"] = df_plot["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

col1, col2 = st.columns(2)

with col1:
  fig_box_age = px.box(
      df_plot,
      x="stroke_label",
      y="age",
      title="뇌졸중 여부별 나이 분포",
      color="stroke_label",
  )
  fig_box_age.update_xaxis(title="뇌졸중 여부")
  fig_box_age.update_yaxis(title="나이")
  st.plotly_chart(fig_box_age, use_container_width=True)

with col2:
  fig_box_glucose = px.box(
      df_plot,
      x="stroke_label",
      y="avg_glucose_level",
      title="뇌졸중 여부별 평균 혈당 분포",
      color="stroke_label",
  )
  fig_box_glucose.update_xaxis(title="뇌졸중 여부")
  fig_box_glucose.update_yaxis(title="평균 혈당")
  st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표
st.markdown("##### 📌 뇌졸중 여부에 따른 변수별 평균값")
mean_df = df.groupby("stroke")[["age", "avg_glucose_level"]].mean().reset_index()
mean_df["stroke"] = mean_df["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})
mean_df.columns = ["뇌졸중 여부", "평균 나이", "평균 혈당"]
st.dataframe(mean_df, use_container_width=True)

st.divider()


# 3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율 막대그래프
st.subheader("3. 기저질환(고혈압, 심장병)에 따른 뇌졸중 비율")
col1, col2 = st.columns(2)

with col1:
  hyp_grouped = (
      df.groupby("hypertension")["stroke"]
      .agg(
          전체인원="count",
          뇌졸중인원=lambda x: (x == 1).sum(),
          발생비율=lambda x: (x == 1).mean() * 100,
      )
      .reset_index()
  )
  hyp_grouped["hypertension"] = hyp_grouped["hypertension"].map(
      {0: "고혈압 없음", 1: "고혈압 있음"}
  )

  fig_hyp = px.bar(
      hyp_grouped,
      x="hypertension",
      y="발생비율",
      title="고혈압 여부에 따른 뇌졸중 비율 (%)",
      text_auto=".2f",
      color="hypertension",
  )
  fig_hyp.update_yaxis(title="뇌졸중 비율 (%)")
  st.plotly_chart(fig_hyp, use_container_width=True)

with col2:
  heart_grouped = (
      df.groupby("heart_disease")["stroke"]
      .agg(
          전체인원="count",
          뇌졸중인원=lambda x: (x == 1).sum(),
          발생비율=lambda x: (x == 1).mean() * 100,
      )
      .reset_index()
  )
  heart_grouped["heart_disease"] = heart_grouped["heart_disease"].map(
      {0: "심장병 없음", 1: "심장병 있음"}
  )

  fig_heart = px.bar(
      heart_grouped,
      x="heart_disease",
      y="발생비율",
      title="심장병 여부에 따른 뇌졸중 비율 (%)",
      text_auto=".2f",
      color="heart_disease",
  )
  fig_heart.update_yaxis(title="뇌졸중 비율 (%)")
  st.plotly_chart(fig_heart, use_container_width=True)

st.divider()


# 4. 체질량지수(bmi) 결측치 분석 및 비율 비교 표
st.subheader("4. 체질량지수(bmi) 결측치 그룹과 전체 데이터 비교")

bmi_missing_df = df[df["bmi"].isnull()]
missing_count = len(bmi_missing_df)
missing_stroke_ratio = (
    (bmi_missing_df["stroke"] == 1).mean() * 100 if missing_count > 0 else 0
)

overall_count = len(df)
overall_stroke_ratio = (df["stroke"] == 1).mean() * 100

comparison_data = pd.DataFrame({
    "구분": ["BMI 결측치 그룹", "전체 데이터"],
    "사람 수": [missing_count, overall_count],
    "뇌졸중 발생 비율 (%)": [
        f"{missing_stroke_ratio:.2f}%",
        f"{overall_stroke_ratio:.2f}%",
    ],
})
st.dataframe(comparison_data, use_container_width=True)

st.divider()


# 5. 흡연 상태(smoking_status)별 인원수 표
st.subheader("5. 흡연 상태(smoking_status)별 인원수")
smoking_counts = df["smoking_status"].value_counts().reset_index()
smoking_counts.columns = ["흡연 상태", "사람 수"]
st.dataframe(smoking_counts, use_container_width=True)
