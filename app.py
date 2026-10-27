
import json
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#17365D"
RED = "#C62828"
LIGHT_BLUE = "#83B4D8"

st.set_page_config(
    page_title="IEEE AI Research Application",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# APPLICATION STYLING
# ============================================================

st.markdown("""
<style>
html, body, [class*="css"], [data-testid="stApp"] {
    font-family: Arial, sans-serif;
}
h1, h2, h3, h4, label {
    font-weight: 700 !important;
}
h1, h2, h3 {
    color: #17365D !important;
}
div.stButton > button[kind="primary"],
div.stDownloadButton > button {
    background: #17365D;
    color: white;
    border: 1px solid #17365D;
    font-weight: 700;
}
div[data-testid="stMetricValue"] {
    color: #17365D;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND APPLICATION CONFIGURATION
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(
        os.path.join(BASE, "model.joblib")
    )
    encoder = joblib.load(
        os.path.join(BASE, "encoder.joblib")
    )
    return model, encoder

@st.cache_data
def load_config():
    with open(
        os.path.join(BASE, "config.json"),
        encoding="utf-8"
    ) as f:
        return json.load(f)

    # Configuration contains feature specifications
    # and model prediction settings.

model, encoder = load_model()
config = load_config()

features = config["features"]
columns = config["feature_order"]
classes = [str(x) for x in encoder.classes_]

positive_class = config["positive_class"]
threshold = config["threshold"]

# ============================================================
# BASIC HELPER FUNCTIONS
# ============================================================

def cast_value(value, feature):
    kind = feature["data_type"]

    if pd.isna(value):
        return np.nan

    if kind == "integer":
        number = float(value)
        if not np.isfinite(number) or not number.is_integer():
            raise ValueError(
                f"Invalid integer for {feature['name']}"
            )
        return int(number)

    if kind == "float":
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(
                f"Invalid numeric value for {feature['name']}"
            )
        return number

    return str(value)


def prepare_dataframe(df):
    missing = [
        col for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing input columns: " +
            ", ".join(missing)
        )

    data = df[columns].copy()

    for feat in features:
        name = feat["name"]
        data[name] = data[name].map(
            lambda v, f=feat: cast_value(v, f)
        )

    for feat in features:
        if feat["input_type"] == "select":
            name = feat["name"]
            allowed = {str(v) for v in feat["choices"]}
            observed = {
                str(v) for v in data[name].dropna().unique()
            }
            unknown = observed - allowed

            if unknown:
                if feat["data_type"] == "integer":
                    for value in unknown:
                        try:
                            number = float(value)
                            if (
                                not np.isfinite(number)
                                or not number.is_integer()
                            ):
                                raise ValueError(
                                    "Invalid categorical integer"
                                )
                        except (ValueError, TypeError):
                            raise ValueError(
                                f"Invalid integer category for {name}"
                            )
                else:
                    raise ValueError(
                        f"Invalid category for {name}: "
                        f"{sorted(unknown)}"
                    )

    return data


def predict_frame(df):
    prepared = prepare_dataframe(df)

    proba = model.predict_proba(prepared)

    if positive_class is None:
        encoded = np.argmax(proba, axis=1)
    else:
        pos_index = classes.index(
            str(positive_class)
        )
        encoded = np.where(
            proba[:, pos_index] >= threshold,
            pos_index,
            1 - pos_index
        )

    predictions = encoder.inverse_transform(
        encoded
    )

    output = pd.DataFrame({
        "Prediction": predictions
    })

    for i, label in enumerate(classes):
        output[
            "Probability_" + label
        ] = proba[:, i]

    return output


def probability_chart(probabilities):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold"
    })

    fig, ax = plt.subplots(figsize=(6, 2.8))

    vals = np.asarray(probabilities, dtype=float)

    colors = [
        RED if str(classes[i]) == str(positive_class)
        else BLUE
        for i in range(len(classes))
    ]

    if positive_class is None:
        colors = [
            BLUE, RED, LIGHT_BLUE
        ][:len(classes)]

    bars = ax.bar(
        classes,
        vals * 100,
        color=colors,
        width=0.55
    )

    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(103, value * 100 + 2),
            f"{value * 100:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=10
        )

    ax.set_ylim(0, 112)
    ax.set_ylabel(
        "Predicted Probability (%)",
        fontweight="bold"
    )

    for label in (
        ax.get_xticklabels() +
        ax.get_yticklabels()
    ):
        label.set_fontweight("bold")

    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.tight_layout()

    return fig


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(config["title"])
st.markdown(
    "**" + config["subtitle"] + "**"
)
st.write(config["description"])
st.caption(
    "Frozen model | Research-only application | "
    f"{len(columns)} input features"
)

st.warning(config["warning"])

tab1, tab2, tab3 = st.tabs([
    "Individual Prediction",
    "Batch CSV Prediction",
    "Model Explainability"
])

# ============================================================
# TAB 1: INDIVIDUAL PREDICTION
# ============================================================

with tab1:

    st.subheader("Enter Input Features")

    user_values = {}

    with st.form("prediction_form"):

        sections = sorted(set(
            f["section"] for f in features
        ))

        for section in sections:

            with st.expander(
                section,
                expanded=(len(sections) == 1)
            ):

                section_features = [
                    f for f in features
                    if f["section"] == section
                ]

                c1, c2 = st.columns(2)

                for i, feat in enumerate(section_features):

                    name = feat["name"]

                    with (c1 if i % 2 == 0 else c2):

                        if feat["input_type"] == "select":

                            choices = feat["choices"]

                            default = feat["default"]

                            try:
                                index = choices.index(default)
                            except ValueError:
                                index = 0

                            selected = st.selectbox(
                                name,
                                choices,
                                index=index,
                                key="input_" + name
                            )

                            user_values[name] = selected

                        else:

                            default = feat["default"]

                            if feat["data_type"] == "integer":
                                selected = st.number_input(
                                    name,
                                    value=int(default),
                                    step=1,
                                    key="input_" + name
                                )
                            else:
                                selected = st.number_input(
                                    name,
                                    value=float(default),
                                    step=0.1,
                                    format="%.4f",
                                    key="input_" + name
                                )

                            user_values[name] = selected

        submit = st.form_submit_button(
            "Generate Prediction",
            type="primary"
        )

    if submit:

        try:

            input_df = pd.DataFrame(
                [user_values]
            )

            result = predict_frame(input_df)

            prediction = str(
                result.iloc[0]["Prediction"]
            )

            prob_cols = [
                "Probability_" + c
                for c in classes
            ]

            scores = (
                result.loc[0, prob_cols]
                .to_numpy(dtype=float)
            )

            st.subheader("Prediction Result")

            left, right = st.columns(2)

            with left:
                st.metric(
                    "Predicted Class",
                    prediction
                )

            with right:

                if positive_class is not None:

                    idx = classes.index(
                        str(positive_class)
                    )

                    st.metric(
                        "Risk Score",
                        f"{scores[idx] * 100:.2f}%"
                    )

                    st.caption(
                        f"Frozen classification "
                        f"threshold: {threshold:.2f}"
                    )

                else:
                    st.metric(
                        "Highest Class Probability",
                        f"{max(scores) * 100:.2f}%"
                    )

            fig = probability_chart(scores)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "These are model-estimated class "
                "probabilities, not guarantees."
            )

        except Exception as error:
            st.error(
                "Prediction could not be generated: "
                + str(error)
            )

# ============================================================
# TAB 2: BATCH CSV PREDICTION
# ============================================================

with tab2:

    st.subheader("Batch Prediction")

    st.write(
        "Upload a CSV file containing the exact "
        "model input feature columns."
    )

    template_row = {
        f["name"]: f["default"]
        for f in features
    }

    template = pd.DataFrame([template_row])

    st.download_button(
        "Download Input CSV Template",
        template.to_csv(index=False),
        file_name="input_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded is not None:

        try:
            uploaded_df = pd.read_csv(uploaded)

            if len(uploaded_df) > 5000:
                raise ValueError(
                    "Maximum batch size is 5000 rows."
                )

            st.write(
                "Uploaded records:",
                len(uploaded_df)
            )

            if st.button(
                "Run Batch Predictions",
                type="primary"
            ):

                output = predict_frame(uploaded_df)

                st.dataframe(
                    output,
                    use_container_width=True
                )

                st.download_button(
                    "Download Prediction Results",
                    output.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

        except Exception as error:
            st.error(str(error))

# ============================================================
# TAB 3: GLOBAL MODEL EXPLAINABILITY
# ============================================================

with tab3:

    st.subheader("Global Feature Importance")

    st.write(
        "This chart shows which encoded features "
        "the trained XGBoost model uses most strongly."
    )

    try:

        transformer = model.named_steps[
            "preprocessing"
        ]

        classifier = model.named_steps[
            "classifier"
        ]

        names = (
            transformer.get_feature_names_out()
        )

        values = (
            classifier.feature_importances_
        )

        importance = pd.DataFrame({
            "Feature": names,
            "Importance": values
        }).sort_values(
            "Importance",
            ascending=False
        )

        top = importance.head(12).iloc[::-1]

        fig, ax = plt.subplots(figsize=(7, 5))

        colors = [
            BLUE if i % 2 == 0 else RED
            for i in range(len(top))
        ]

        ax.barh(
            top["Feature"],
            top["Importance"],
            color=colors
        )

        ax.set_xlabel(
            "XGBoost Feature Importance",
            fontweight="bold"
        )

        for label in (
            ax.get_xticklabels() +
            ax.get_yticklabels()
        ):
            label.set_fontweight("bold")

        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Download Feature Importance",
            importance.to_csv(index=False),
            file_name="feature_importance.csv",
            mime="text/csv"
        )

        st.info(
            "Global feature importance is not a "
            "personalized explanation or a causal "
            "interpretation. Class-specific SHAP "
            "analyses are available in the "
            "research results."
        )

    except Exception as error:
        st.error(
            "Unable to display feature importance: "
            + str(error)
        )

st.divider()

st.caption(
    "IEEE Research Prototype | "
    "Red–Blue Scientific Interface | "
    "Model configuration frozen after training"
)
