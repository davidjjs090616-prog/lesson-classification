    glucose_fig.update_layout(yaxis_title="사람 수", xaxis_title="평균 혈당")
    st.plotly_chart(glucose_fig, use_container_width=True)

st.divider()
st.header("2. 뇌졸중 경험 여부에 따른 비교")
st.write("뇌졸중을 겪지 않은 사람과 겪은 사람의 나이 및 평균 혈당 분포를 상자그림으로 비교합니다.")

box_left, box_right = st.columns(2)
box_data = plot_df.dropna(subset=["뇌졸중 여부"])

with box_left:
    age_box = px.box(
        box_data.dropna(subset=["age"]),
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]},
        title="뇌졸중 경험 여부별 나이",
        labels={"age": "나이", "뇌졸중 여부": "그룹"},
    )
    age_box.update_layout(showlegend=False, xaxis_title="그룹", yaxis_title="나이")
    st.plotly_chart(age_box, use_container_width=True)

with box_right:
    glucose_box = px.box(
        box_data.dropna(subset=["avg_glucose_level"]),
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]},
        title="뇌졸중 경험 여부별 평균 혈당",
        labels={"avg_glucose_level": "평균 혈당", "뇌졸중 여부": "그룹"},
    )
    glucose_box.update_layout(showlegend=False, xaxis_title="그룹", yaxis_title="평균 혈당")
    st.plotly_chart(glucose_box, use_container_width=True)

mean_table = (
    box_data.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .reset_index()
    .rename(columns={
        "뇌졸중 여부": "그룹",
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당의 평균",
    })
)
mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당의 평균"] = mean_table["평균 혈당의 평균"].round(2)
st.subheader("두 그룹의 평균값")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

st.divider()
st.header("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")
st.write("각 위험 요인이 없는 그룹과 있는 그룹에서 stroke가 1인 비율을 비교합니다.")

risk_left, risk_right = st.columns(2)

with risk_left:
    hypertension_table = make_binary_rate_table(plot_df, "hypertension")
    hypertension_fig = px.bar(
        hypertension_table,
        x="구분",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="고혈압 여부별 뇌졸중 비율",
        labels={"구분": "고혈압", "뇌졸중 비율(%)": "뇌졸중 비율(%)"},
    )
    hypertension_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    hypertension_fig.update_layout(yaxis_title="뇌졸중 비율(%)", xaxis_title="고혈압")
    st.plotly_chart(hypertension_fig, use_container_width=True)

with risk_right:
    heart_table = make_binary_rate_table(plot_df, "heart_disease")
    heart_fig = px.bar(
        heart_table,
        x="구분",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="심장병 여부별 뇌졸중 비율",
        labels={"구분": "심장병", "뇌졸중 비율(%)": "뇌졸중 비율(%)"},
    )
    heart_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    heart_fig.update_layout(yaxis_title="뇌졸중 비율(%)", xaxis_title="심장병")
    st.plotly_chart(heart_fig, use_container_width=True)

st.divider()
st.header("4. BMI 빈 값과 뇌졸중 비율")
st.write("체질량지수(bmi)가 비어 있는 사람과 전체 사람의 뇌졸중 비율을 나란히 비교합니다.")

missing_bmi_people = df[df["bmi"].isna()]
comparison_table = pd.DataFrame([
    {
        "대상": "BMI가 비어 있는 사람",
        "사람 수": len(missing_bmi_people),
        "stroke가 1인 사람 수": int((missing_bmi_people["stroke"] == 1).sum()),
        "뇌졸중 비율(%)": stroke_rate_percent(missing_bmi_people["stroke"]),
    },
    {
        "대상": "전체 사람",
        "사람 수": len(df),
        "stroke가 1인 사람 수": int((df["stroke"] == 1).sum()),
        "뇌졸중 비율(%)": stroke_rate_percent(df["stroke"]),
    },
])
comparison_table["뇌졸중 비율(%)"] = comparison_table["뇌졸중 비율(%)"].round(2)
st.dataframe(comparison_table, use_container_width=True, hide_index=True)

st.divider()
st.header("5. 흡연 상태별 사람 수")
st.write("흡연 상태의 값마다 해당하는 사람 수를 셉니다. 빈 값이 있다면 '빈 값'으로 표시합니다.")
smoking_values = df["smoking_status"].astype("object").where(
    df["smoking_status"].notna(), "빈 값"
)
smoking_table = (
    smoking_values.value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_table, use_container_width=True, hide_index=True)

st.divider()
st.caption(f"데이터 파일: {DATA_URL} · CSV 인코딩: UTF-8")
