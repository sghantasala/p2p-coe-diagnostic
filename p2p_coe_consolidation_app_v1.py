
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests
import json

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(
    page_title="P2P CoE Diagnostic – Consolidation & Recommendations",
    layout="wide"
)

APP_TITLE = "P2P CoE Diagnostic – Consolidation Dashboard"
APP_SUBTITLE = "StrategyStack Consulting | Multi-Stakeholder Maturity View + AI Recommendations"

# ---------------------------
# OPENAI HELPER
# ---------------------------

def call_openai_chat(prompt: str) -> str:
    """Low-level OpenAI Chat Completions call. Returns raw text or error message."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: Missing OPENAI_API_KEY environment variable."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an experienced management consultant and P2P CoE expert. "
                    "You write concise, CxO-ready recommendations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "ERROR: Timeout contacting OpenAI API. Please try again (network was too slow)."
    except Exception as e:
        return f"ERROR: {e}"


def generate_ai_recommendations(dim_scores_ai: dict, dim_scores_final: dict) -> str:
    """Generate consolidated executive summary and recommendations based on aggregated scores."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "AI recommendations not generated – missing OPENAI_API_KEY.\n"
            "Set your OpenAI key as an environment variable or in Streamlit secrets and rerun the app."
        )

    people_ai = dim_scores_ai.get("People", 0)
    process_ai = dim_scores_ai.get("Process", 0)
    tech_ai = dim_scores_ai.get("Technology", 0)

    people_final = dim_scores_final.get("People", people_ai)
    process_final = dim_scores_final.get("Process", process_ai)
    tech_final = dim_scores_final.get("Technology", tech_ai)

    prompt = f"""
You are reviewing a consolidated P2P (Procure-to-Pay) maturity assessment across multiple stakeholders.
Scores are on a 1–5 scale (1 = very poor, 5 = world-class).

AI Average Scores (pre-consolidation):
- People: {people_ai:.2f}
- Process: {process_ai:.2f}
- Technology: {tech_ai:.2f}

Consultant Final Consolidated Scores:
- People: {people_final:.2f}
- Process: {process_final:.2f}
- Technology: {tech_final:.2f}

Using this information, produce a concise, CxO-ready output with the following sections:

1) Executive Summary
   - 3–5 bullet points capturing the overall maturity and big-picture story.

2) Key Gaps & Pain Points
   - 5–7 bullets grouped under People, Process, and Technology.
   - Focus on structural and recurring issues, not one-off comments.

3) Quick-Win Recommendations (0–3 months)
   - 5–7 specific actions the client can take quickly with high impact and low effort.
   - Mix of policy, process, training, and basic system tweaks.

4) Medium-Term CoE Build-Out Initiatives (6–18 months)
   - 4–6 initiatives that build a sustainable P2P CoE (e.g., governance, automation, data, operating model).
   - Each initiative should be 1–2 lines, outcome-oriented.

Keep the tone consulting-grade, practical, and suitable for inclusion in a client-facing deck.
Do NOT restate the numeric scores; interpret them.
"""

    raw = call_openai_chat(prompt)
    return raw

# ---------------------------
# BRANDING HELPERS
# ---------------------------

def render_top_branding():
    col_logo1, col_title, col_logo2 = st.columns([1, 6, 1])

    with col_logo1:
        if os.path.exists("ssc_logo.png"):
            st.image("ssc_logo.png", width=110)
        else:
            st.markdown("**SSC**")

    with col_title:
        st.markdown(
            f"<h1 style='text-align:center;color:#002060;font-family:Segoe UI, sans-serif;'>{APP_TITLE}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h4 style='text-align:center;color:gray;font-family:Segoe UI, sans-serif;'>{APP_SUBTITLE}</h4>",
            unsafe_allow_html=True,
        )

    with col_logo2:
        if os.path.exists("bain_logo.png"):
            st.image("bain_logo.png", width=110)
        else:
            st.markdown("**Client**")

    st.markdown("---")


def render_sidebar_branding():
    st.sidebar.markdown("### StrategyStack Consulting")
    if os.path.exists("ssc_logo.png"):
        st.sidebar.image("ssc_logo.png", width=160)
    st.sidebar.markdown("P2P CoE Consolidation")
    st.sidebar.markdown("---")

# ---------------------------
# MAIN LOGIC
# ---------------------------

def main():
    render_top_branding()
    render_sidebar_branding()

    st.markdown("#### Step 1: Upload Question-Level CSV Files from Individual Assessments")
    st.write(
        "Upload one or more CSV files exported from the **Consultant Review & Dashboard** tab "
        "of the P2P CoE Diagnostic App (V4). Each file should contain columns like: "
        "`dimension, index, question, answer, ai_score, final_score, reason`."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one CSV file to begin consolidation.")
        return

    # Combine all files
    all_dfs = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Could not read file {f.name}: {e}")

    if not all_dfs:
        st.error("No valid CSV files could be read. Please check the format and try again.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Basic validation
    required_cols = {"dimension", "index", "question", "answer", "ai_score", "final_score", "reason"}
    missing = required_cols - set(df_all.columns)
    if missing:
        st.error(f"The uploaded files are missing required columns: {missing}")
        st.stop()

    st.success(f"Loaded {len(df_all)} rows from {len(uploaded_files)} file(s).")

    with st.expander("Preview Combined Question-Level Data", expanded=False):
        st.dataframe(df_all.head(50))

    # ---------------------------
    # AGGREGATION
    # ---------------------------

    st.markdown("#### Step 2: Aggregated View by Dimension & Question")

    # Aggregate per dimension + question
    agg_q = (
        df_all
        .groupby(["dimension", "index", "question"], as_index=False)
        .agg(
            n_responses=("final_score", "count"),
            avg_ai_score=("ai_score", "mean"),
            avg_final_score=("final_score", "mean"),
        )
    )

    st.dataframe(agg_q)

    # Dimension-level averages
    agg_dim = (
        df_all
        .groupby("dimension", as_index=False)
        .agg(
            avg_ai_score=("ai_score", "mean"),
            avg_final_score=("final_score", "mean"),
            n_responses=("final_score", "count"),
        )
    )

    st.markdown("#### Step 3: Dimension-Level Consolidated Scores")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**AI Average Scores (from all responses)**")
        st.dataframe(agg_dim[["dimension", "avg_ai_score", "n_responses"]])

    with col2:
        st.write("**Consultant Average Scores (from all responses)**")
        st.dataframe(agg_dim[["dimension", "avg_final_score", "n_responses"]])

    # Build dictionaries for dimensions
    dim_scores_ai = {
        row["dimension"]: row["avg_ai_score"]
        for _, row in agg_dim.iterrows()
    }
    dim_scores_final_default = {
        row["dimension"]: row["avg_final_score"]
        for _, row in agg_dim.iterrows()
    }

    # ---------------------------
    # CONSULTANT OVERRIDE (DIMENSION-LEVEL)
    # ---------------------------

    st.markdown("#### Step 4: Consultant Override – Final Dimension Maturity Scores")

    dim_scores_final = {}
    for dim in ["People", "Process", "Technology"]:
        ai_val = float(dim_scores_ai.get(dim, 0.0))
        default_final = float(dim_scores_final_default.get(dim, ai_val))
        st.write(f"**{dim}**")
        final_val = st.slider(
            f"{dim} – Final Consolidated Score (1–5)",
            min_value=1.0,
            max_value=5.0,
            value=float(round(default_final if default_final > 0 else 3.0)),
            step=0.1,
        )
        dim_scores_final[dim] = final_val

    # Visuals
    st.markdown("#### Step 5: Visualize Consolidated Maturity")

    radar_df = pd.DataFrame({
        "Dimension": ["People", "Process", "Technology"],
        "AI Score": [
            dim_scores_ai.get("People", 0.0),
            dim_scores_ai.get("Process", 0.0),
            dim_scores_ai.get("Technology", 0.0),
        ],
        "Final Score": [
            dim_scores_final.get("People", 0.0),
            dim_scores_final.get("Process", 0.0),
            dim_scores_final.get("Technology", 0.0),
        ],
    })

    # Radar for final scores
    fig_radar = px.line_polar(
        radar_df,
        r="Final Score",
        theta="Dimension",
        line_close=True,
        range_r=[0, 5],
        title="Consolidated Final Maturity (Consultant View)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Bar chart AI vs Final
    fig_bar = px.bar(
        radar_df.melt(id_vars="Dimension", value_vars=["AI Score", "Final Score"], var_name="Type", value_name="Score"),
        x="Dimension",
        y="Score",
        color="Type",
        barmode="group",
        range_y=[0, 5],
        title="AI vs Consultant Final Scores by Dimension"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------
    # AI RECOMMENDATIONS
    # ---------------------------

    st.markdown("#### Step 6: Generate AI Recommendations (CxO-Ready)")
    if st.button("Generate Consolidated AI Recommendations"):
        with st.spinner("Calling AI to generate executive summary and recommendations..."):
            recs = generate_ai_recommendations(dim_scores_ai, dim_scores_final)
        st.markdown("##### AI-Generated Narrative")
        if recs.startswith("ERROR:"):
            st.error(recs)
        else:
            st.write(recs)

    # ---------------------------
    # EXPORT CONSOLIDATED DATA
    # ---------------------------

    st.markdown("#### Step 7: Export Consolidated Data")
    st.write("Download the combined question-level dataset for offline analysis or archival.")

    csv_all = df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Combined Question-Level CSV",
        data=csv_all,
        file_name="p2p_coe_consolidated_question_level.csv",
        mime="text/csv"
    )

    csv_agg_q = agg_q.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Aggregated Question-Level CSV",
        data=csv_agg_q,
        file_name="p2p_coe_aggregated_question_level.csv",
        mime="text/csv"
    )

    csv_agg_dim = agg_dim.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Aggregated Dimension-Level CSV",
        data=csv_agg_dim,
        file_name="p2p_coe_aggregated_dimension_level.csv",
        mime="text/csv"
    )

    # Footer
    st.markdown(
        """
        <hr>
        <div style='text-align:center;color:gray;font-size:12px;'>
        P2P CoE Consolidation App • StrategyStack Consulting © 2025<br>
        Multi-stakeholder consolidation of P2P maturity for CoE design
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
