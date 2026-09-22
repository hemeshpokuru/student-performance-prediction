import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Performance Prediction")

st.markdown(
    """
    Predict a student's **Performance Index** based on academic
    performance, study habits, sleep, extracurricular activities,
    and sample question paper practice.
    """
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📊 Final Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "R² Score",
    "0.9879"
)

col2.metric(
    "RMSE",
    "2.1239"
)

col3.metric(
    "MAE",
    "1.6803"
)

col4.metric(
    "CV R²",
    "0.9880"
)

st.caption(
    "Final model: Tuned Gradient Boosting Regression"
)

st.divider()


# ============================================================
# STUDENT PREDICTION
# ============================================================

st.header("👨‍🎓 Predict Student Performance")

st.write(
    "Enter the student's details below to estimate the Performance Index."
)

col1, col2 = st.columns(2)

with col1:

    hours_studied = st.number_input(
        "Hours Studied",
        min_value=1,
        max_value=9,
        value=5,
        step=1
    )

    previous_scores = st.number_input(
        "Previous Scores",
        min_value=40,
        max_value=99,
        value=70,
        step=1
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )


with col2:

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=4,
        max_value=9,
        value=7,
        step=1
    )

    sample_papers = st.number_input(
        "Sample Question Papers Practiced",
        min_value=0,
        max_value=9,
        value=5,
        step=1
    )


if st.button(
    "🎯 Predict Performance Index",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Hours Studied": [hours_studied],
        "Previous Scores": [previous_scores],
        "Extracurricular Activities": [extracurricular],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers]
    })

    prediction = model.predict(input_data)[0]

    prediction = np.clip(prediction, 10, 100)

    # Store prediction in session history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    st.session_state.prediction_history.append({
        "Hours Studied": hours_studied,
        "Previous Scores": previous_scores,
        "Extracurricular Activities": extracurricular,
        "Sleep Hours": sleep_hours,
        "Sample Papers": sample_papers,
        "Predicted Performance": round(prediction, 2)
    })

    st.success(
        f"🎯 Predicted Performance Index: **{prediction:.2f}**"
    )


# ============================================================
# PREDICTION SUMMARY
# ============================================================

st.divider()

st.header("📋 Prediction Summary")

if "prediction_history" in st.session_state and st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No predictions have been made yet. "
        "Enter student details above and click Predict."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header("📈 Feature Importance")

st.write(
    "The following feature importance values were obtained "
    "from the feature-importance analysis performed during model development."
)

feature_importance = pd.DataFrame({
    "Feature": [
        "Previous Scores",
        "Hours Studied",
        "Sleep Hours",
        "Sample Question Papers Practiced",
        "Extracurricular Activities"
    ],
    "Importance": [
        1.686443,
        0.294443,
        0.003649,
        0.001803,
        0.000622
    ]
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=True
)

st.bar_chart(
    feature_importance.set_index("Feature")
)

st.dataframe(
    feature_importance.sort_values(
        "Importance",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.header("📊 Model Comparison")

model_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        1.646970,
        2.433249,
        1.895022,
        1.695763
    ],
    "MSE": [
        4.305901,
        9.306132,
        5.615141,
        4.575901
    ],
    "RMSE": [
        2.075066,
        3.050595,
        2.369629,
        2.139136
    ],
    "R²": [
        0.988430,
        0.974995,
        0.984912,
        0.987705
    ]
})

st.dataframe(
    model_results,
    use_container_width=True,
    hide_index=True
)

st.write("### Model R² Comparison")

st.bar_chart(
    model_results.set_index("Model")["R²"]
)


# ============================================================
# HYPERPARAMETER TUNING RESULTS
# ============================================================

st.divider()

st.header("⚙️ Hyperparameter Tuning")

st.write(
    "The Gradient Boosting model was tuned using cross-validation."
)

tuning_results = {
    "Learning Rate": 0.1,
    "Maximum Depth": 3,
    "Minimum Samples Leaf": 1,
    "Minimum Samples Split": 2,
    "Number of Estimators": 150,
    "Cross-Validation R²": 0.987953
}

for key, value in tuning_results.items():

    st.write(
        f"**{key}:** {value}"
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.divider()

st.header("💡 Business & Data-Driven Insights")

st.markdown(
    """
    ### 1. Previous Scores are the strongest factor

    Previous Scores show the strongest relationship with the
    Performance Index, with a correlation of approximately **0.92**.

    This indicates that a student's previous academic performance
    is strongly associated with their expected current performance.

    ### 2. Hours Studied has a positive relationship

    Hours Studied has a correlation of approximately **0.37**
    with Performance Index.

    Students who spend more time studying generally tend to have
    higher predicted performance.

    ### 3. Previous academic performance is more informative than study time

    Although study time is associated with performance, the correlation
    analysis shows that Previous Scores have a considerably stronger
    relationship with Performance Index.

    ### 4. Sleep Hours show a weak direct relationship

    Sleep Hours have a correlation of approximately **0.05**
    with Performance Index in this dataset.

    Therefore, within this particular dataset, sleep duration alone
    provides limited predictive information.

    ### 5. Sample Question Papers have a weak direct relationship

    Sample Question Papers Practiced have a correlation of approximately
    **0.04** with Performance Index.

    Their individual predictive contribution is relatively small
    compared with Previous Scores and Hours Studied.

    ### 6. Extracurricular Activities have limited predictive contribution

    The dataset contains a nearly balanced distribution:

    - **Yes:** 4,948 students
    - **No:** 5,052 students

    The analysis indicates that extracurricular participation alone
    has very limited predictive contribution.

    ### 7. The model provides highly accurate predictions

    The tuned Gradient Boosting model achieved approximately:

    - **R² = 0.9879**
    - **RMSE = 2.1239**
    - **MAE = 1.6803**

    This indicates that the model predictions are close to the
    actual Performance Index values in the test dataset.
    """
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.header("ℹ️ About the Project")

st.markdown(
    """
    **Objective**

    Develop a Machine Learning regression model that predicts
    student Performance Index based on academic and lifestyle factors.

    **Dataset**

    - Records: 10,000
    - Features: 5
    - Target: Performance Index
    - Target range: 10–100

    **Machine Learning Models**

    - Linear Regression
    - Decision Tree Regression
    - Random Forest Regression
    - Gradient Boosting Regression

    **Final Model**

    Tuned Gradient Boosting Regression.

    **Evaluation Metrics**

    - MAE
    - MSE
    - RMSE
    - R² Score
    - Cross-Validation R²
    """
)

st.caption(
    "Student Performance Prediction | Machine Learning Regression Project"
)
