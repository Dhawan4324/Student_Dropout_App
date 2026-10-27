import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
BLUE = "#17365D"
RED = "#C62828"
PALE_BLUE = "#EAF2FA"
LIGHT_BLUE = "#83B4D8"

st.set_page_config(
    page_title="Student Academic Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --navy:#17365D; --red:#C62828; }
.stApp { background-color:#F6F8FC; }
.block-container { max-width:1440px; padding-top:1.4rem; padding-bottom:3rem; }
h1,h2,h3,h4, label { font-weight:750 !important; color:#17365D !important; }
[data-testid="stSidebar"] { background:#EAF2FA; border-right:1px solid #D8E3F0; }
[data-testid="stSidebar"] h2 { color:#17365D; }
[data-testid="stMetric"] { border:1px solid #DBE6F2; background:white;
    padding:13px 17px; border-radius:12px; }
[data-testid="stMetricValue"] { font-weight:800; color:#17365D; }
[data-testid="stExpander"] { background:white; border:1px solid #D9E3EF;
    border-radius:12px; margin-bottom:10px; }
[data-testid="stExpander"] summary { font-weight:750; color:#17365D; }
div.stButton > button[kind="primary"], 
div.stFormSubmitButton > button[kind="primary"] {
    background:#C62828; border:1px solid #C62828; color:white;
    font-weight:800; border-radius:9px; min-height:45px;
}
div.stButton > button[kind="primary"]:hover,
div.stFormSubmitButton > button[kind="primary"]:hover {
    background:#A71F1F; border-color:#A71F1F; color:white;
}
.stTabs [data-baseweb="tab"] { font-weight:750; }
.hero { padding:22px 26px; border-radius:14px;
    background:linear-gradient(115deg,#17365D,#275D91); color:white;
    margin-bottom:15px; }
.hero h1 { color:white !important; font-size:1.72rem; margin:0 0 4px 0; }
.hero p { margin:0; color:#E9F2FC; font-size:.96rem; }
.mini-card { border:1px solid #DDE6F1; background:white;
    border-radius:12px; padding:13px 17px; min-height:100px; }
.mini-card .card-title { font-size:1.04rem; font-weight:800; margin-bottom:5px; }
.mini-card .card-text { font-size:.82rem; color:#475569; line-height:1.4; }
.panel { background:white; border-radius:12px; border:1px solid #DDE6F1;
    padding:16px 18px; margin-bottom:13px; }
.panel-title { font-size:.96rem; font-weight:800; color:#17365D; margin-bottom:6px; }
.panel-content { color:#41536C; font-size:.87rem; line-height:1.6; }
.section-kicker { color:#C62828; font-weight:850; letter-spacing:.06em;
    text-transform:uppercase; font-size:.73rem; margin:8px 0; }
@media (max-width:760px) { .hero { padding:17px; } .hero h1 { font-size:1.4rem; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    # Only load these trusted artifacts from your own GitHub repository.
    model = joblib.load(BASE / "model.joblib")
    encoder = joblib.load(BASE / "encoder.joblib")
    return model, encoder


@st.cache_data
def load_config():
    return json.loads((BASE / "config.json").read_text(encoding="utf-8"))


model, encoder = load_artifacts()
config = load_config()
features = config["features"]
columns = config["feature_order"]
classes = [str(x) for x in encoder.classes_]
feature_map = {f["name"]: f for f in features}

assert len(columns) == 36, "Expected 36 trained model inputs"
assert config["positive_class"] is None, "This interface expects multiclass prediction"


def normalize_value(value, feat):
    if pd.isna(value):
        return np.nan
    kind = feat["data_type"]
    if kind == "integer":
        number = float(value)
        if not np.isfinite(number) or not number.is_integer():
            raise ValueError(f"{feat['name']}: enter a whole-number code")
        return int(number)
    if kind == "float":
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{feat['name']}: enter a finite number")
        return number
    return str(value)


def prepare_dataframe(data):
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    frame = data.loc[:, columns].copy()
    for name in columns:
        feat = feature_map[name]
        frame[name] = frame[name].map(lambda x, f=feat: normalize_value(x, f))
        if feat["input_type"] == "select" and feat["data_type"] == "string":
            valid = {str(x) for x in feat["choices"]}
            observed = {str(x) for x in frame[name].dropna().unique()}
            if observed - valid:
                raise ValueError(
                    f"{name}: unsupported category {sorted(observed-valid)}"
                )
        # Integer category codes not observed during training are permitted;
        # the saved fitted OneHotEncoder handles unknown/infrequent values.
    return frame


def predict_frame(data):
    prepared = prepare_dataframe(data)
    probabilities = model.predict_proba(prepared)
    predicted = encoder.inverse_transform(probabilities.argmax(axis=1))
    result = pd.DataFrame({"Prediction": predicted})
    for i, label in enumerate(classes):
        result[f"Probability_{label}"] = probabilities[:, i]
    return result


def pretty_name(name):
    name = name.replace("Curricular units ", "Curricular Units ")
    name = name.replace("1st sem", "Sem 1").replace("2nd sem", "Sem 2")
    name = name.replace("(without evaluations)", "(No Evaluations)")
    name = name.replace("Nacionality", "Nationality (Dataset Code)")
    return name


def outcome_card(name, icon, description, accent, tint):
    st.markdown(
        f'<div class="mini-card" style="background:{tint};">'
        f'<div class="card-title" style="color:{accent}">{icon} {name}</div>'
        f'<div class="card-text">{description}</div></div>',
        unsafe_allow_html=True
    )


def probability_figure(probabilities):
    values = np.asarray(probabilities, dtype=float) * 100
    labels = classes
    palette = [RED if x == "Dropout" else (BLUE if x == "Enrolled" else LIGHT_BLUE)
               for x in labels]
    fig, ax = plt.subplots(figsize=(6.0, 2.0))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=palette, height=.48)
    ax.set_yticks(y, labels, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Predicted probability (%)", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    for bar, score in zip(bars, values):
        ax.text(min(score + 2, 103), bar.get_y()+bar.get_height()/2,
                f"{score:.1f}%", va="center", fontsize=9, fontweight="bold",
                color=BLUE)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for edge in ("top","right","left"):
        ax.spines[edge].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.2)
    return fig


with st.sidebar:
    st.markdown("## 🎓 Academic AI")
    st.caption("Explainable research prototype")
    st.divider()
    st.markdown("**Research configuration**")
    st.write("**Model:** XGBoost Classifier")
    st.write("**Input features:** 36")
    st.write("**Output categories:** 3")
    st.write("**Prediction stage:** After Semester 2")
    st.divider()
    st.info("Predictions are for academic decision support, not automatic student decisions.")

st.markdown(
    '<div class="hero"><h1>🎓 Student Academic Outcome Predictor</h1>'
    '<p>Explainable machine learning for student outcome analysis and academic support</p>'
    '</div>',
    unsafe_allow_html=True
)

top1, top2, top3 = st.columns(3, gap="medium")
with top1:
    outcome_card("Dropout", "🔴", "Student may leave the academic program.", RED, "#FFF2F2")
with top2:
    outcome_card("Enrolled", "🔵", "Student may continue studying.", BLUE, "#EAF3FF")
with top3:
    outcome_card("Graduate", "🎓", "Student may complete the program.", "#175C7A", "#ECF7FA")

st.caption(
    "Research model using first- and second-semester records. "
    "It is not an admission-stage prediction tool."
)

tab1, tab2, tab3 = st.tabs(
    ["🎯 Individual Prediction", "📁 Batch CSV Prediction", "📊 Model Explainability"]
)

with tab1:
    main, info = st.columns([3.25, 1.15], gap="large")
    with main:
        st.markdown("### Student Information")
        st.caption("Complete all three academic-information sections before prediction.")

        # All 36 fields remain accessible, in the same names/order as config.json.
        section_order = [
            "First Semester Academic Information",
            "Second Semester Academic Information",
            "Student and Admission Information",
        ]
        section_order += [s for s in dict.fromkeys(f["section"] for f in features)
                          if s not in section_order]

        values = {}
        with st.form("student_prediction_form", clear_on_submit=False):
            for index, section in enumerate(section_order):
                group = [f for f in features if f["section"] == section]
                if not group:
                    continue
                with st.expander(
                    f"{index+1:02d}  {section}",
                    expanded=(index == 0)
                ):
                    row_cols = st.columns(3, gap="medium")
                    for i, feat in enumerate(group):
                        name = feat["name"]
                        with row_cols[i % 3]:
                            if feat["input_type"] == "select":
                                choices = feat["choices"]
                                try:
                                    default_index = choices.index(feat["default"])
                                except ValueError:
                                    default_index = 0
                                value = st.selectbox(
                                    pretty_name(name), choices,
                                    index=default_index,
                                    key=f"field_{name}",
                                    help=f"Original dataset field: {name}",
                                )
                            elif feat["data_type"] == "integer":
                                value = st.number_input(
                                    pretty_name(name), value=int(feat["default"]),
                                    step=1, key=f"field_{name}"
                                )
                            elif feat["data_type"] == "float":
                                value = st.number_input(
                                    pretty_name(name), value=float(feat["default"]),
                                    step=.1, format="%.3f", key=f"field_{name}"
                                )
                            else:
                                value = st.text_input(
                                    pretty_name(name), value=str(feat["default"]),
                                    key=f"field_{name}"
                                )
                            values[name] = value

            submitted = st.form_submit_button(
                "⚡ Generate Academic Outcome Prediction",
                type="primary", use_container_width=True
            )

        if submitted:
            try:
                output = predict_frame(pd.DataFrame([values]))
                label = str(output.loc[0, "Prediction"])
                probs = np.array(
                    [output.loc[0, f"Probability_{c}"] for c in classes],
                    dtype=float
                )
                st.session_state["last_prediction"] = {
                    "label": label, "probabilities": probs.tolist()
                }
            except Exception as exc:
                st.error(f"Unable to generate prediction: {exc}")

        if "last_prediction" in st.session_state:
            data = st.session_state["last_prediction"]
            label = data["label"]
            probs = np.asarray(data["probabilities"], dtype=float)
            st.markdown("### Prediction Results")
            a, b, c = st.columns(3)
            a.metric("Predicted Outcome", label)
            b.metric("Highest Class Probability", f"{100*np.max(probs):.2f}%")
            c.metric("Dropout Class Probability",
                     f"{100*probs[classes.index('Dropout')]:.2f}%" if "Dropout" in classes else "N/A")
            fig = probability_figure(probs)
            # Narrow, publication-friendly chart; not stretched to page width.
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.caption(
                "Probabilities describe model predictions and may not be perfectly "
                "calibrated. They should be interpreted alongside academic context."
            )
    with info:
        st.markdown(
            '<div class="panel"><div class="panel-title">ℹ️ About this application</div>'
            '<div class="panel-content">A multiclass XGBoost research model '
            'predicting Dropout, Enrolled or Graduate outcomes from '
            'academic and student attributes.</div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="panel"><div class="panel-title">🧾 Input features</div>'
            '<div style="font-size:2rem;font-weight:850;color:#C62828">36</div>'
            '<div class="panel-content">Academic and student variables</div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="panel"><div class="panel-title">🧠 ML model</div>'
            '<div class="panel-content"><b>XGBoost Classifier</b><br>'
            'Saved preprocessing and trained prediction model.</div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="panel"><div class="panel-title">📌 Intended use</div>'
            '<div class="panel-content">Academic research, model demonstration, '
            'and supportive review of student outcomes.</div></div>',
            unsafe_allow_html=True
        )

with tab2:
    st.markdown("### Batch Student Outcome Prediction")
    st.write("Upload CSV data containing the same 36 feature columns used during training.")
    template = pd.DataFrame([{f["name"]: f["default"] for f in features}])
    st.download_button(
        "⬇ Download CSV Template",
        template.to_csv(index=False),
        file_name="student_dropout_input_template.csv",
        mime="text/csv"
    )
    upload = st.file_uploader("Choose student records (.csv)", type="csv")
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            st.caption(f"{len(raw)} rows detected")
            if len(raw) > 5000:
                st.error("Maximum batch size: 5,000 rows.")
            elif st.button("Run Batch Predictions", type="primary"):
                output = predict_frame(raw)
                st.dataframe(output, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇ Download Prediction Results",
                    output.to_csv(index=False),
                    "student_dropout_batch_predictions.csv",
                    mime="text/csv"
                )
        except Exception as exc:
            st.error(f"Batch processing failed: {exc}")

with tab3:
    st.markdown("### Global Model Explainability")
    st.info(
        "The chart summarizes the fitted model's global feature importance. "
        "It is not a personal explanation or proof of causal influence."
    )
    try:
        preprocessor = model.named_steps["preprocessing"]
        classifier = model.named_steps["classifier"]
        names = preprocessor.get_feature_names_out()
        scores = classifier.feature_importances_
        imp = pd.DataFrame({"Feature": names, "Importance": scores}).sort_values(
            "Importance", ascending=False
        )
        subset = imp.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.barh(
            subset["Feature"].str.replace(r"^(numeric|categorical|num|cat)__", "", regex=True),
            subset["Importance"],
            color=[BLUE if i % 2 == 0 else RED for i in range(len(subset))]
        )
        ax.set_xlabel("XGBoost feature importance", fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        st.download_button(
            "⬇ Download Feature Importance",
            imp.to_csv(index=False),
            "student_dropout_feature_importance.csv", mime="text/csv"
        )
    except Exception as exc:
        st.error(f"Explainability chart could not be loaded: {exc}")

st.divider()
st.caption(
    "Research prototype · Frozen XGBoost pipeline · "
    "Class outputs: Dropout / Enrolled / Graduate"
)
