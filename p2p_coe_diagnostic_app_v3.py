
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import os
import json

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(
    page_title="P2P CoE Diagnostic - SSC (v4)",
    layout="wide"
)

APP_TITLE = "P2P CoE Diagnostic – People, Process & Technology"
APP_SUBTITLE = "StrategyStack Consulting | Client Self-Assessment + AI Pre-Scoring + Consultant Override"

# ---------------------------
# QUESTION DEFINITIONS
# ---------------------------

PEOPLE_QUESTIONS = [
    "Describe the current P2P organisation structure across entities. Who owns what?",
    "How clearly are job roles and responsibilities defined for P2P-related positions?",
    "How well do procurement, finance/AP, and site/operations teams collaborate in the P2P process?",
    "Where do you see duplication of effort or unclear ownership in P2P today?",
    "Describe current spans of control (manager-to-staff ratios) for P2P teams. Any issues?",
    "Is workload balanced across team members, or are certain roles/people always overloaded?",
    "How would you describe the team’s procurement skills (e.g., category knowledge, negotiation, RFQ)?",
    "How would you describe the AP team’s skills for invoice processing, matching and reconciliation?",
    "What formal training exists for onboarding and upskilling P2P staff (if any)?",
    "How familiar are teams with ERP, workflows, and compliance policies related to P2P?",
    "Has attrition or churn impacted process stability or knowledge retention in P2P?",
    "Do team members clearly understand the P2P KPIs/SLAs they are accountable for?",
    "How often are P2P performance reviews held, and how are results communicated?",
    "Is there cross-training or backup coverage for key P2P roles? Describe gaps if any."
]

PROCESS_QUESTIONS = [
    "Describe how purchase requisitions (PRs) are created today (who, how, and with what quality).",
    "What are the most common issues/errors with PRs (missing data, wrong codes, etc.)?",
    "Is there a standard PR-to-approval process followed consistently across entities? Where does it differ?",
    "How are purchase orders (POs) created and approved? Are there any manual/off-system POs?",
    "How standardized is vendor onboarding (steps, checks, documentation)?",
    "How are vendor validations (KYC, tax IDs, banking, compliance) carried out and recorded?",
    "Describe the process for vendor master creation, updates, and deactivation. Any issues?",
    "Describe how GRNs (goods receipts) are done. Where do delays or gaps typically occur?",
    "How are services (non-materials) receipted and approved today?",
    "Describe how invoices are received (channels), validated, and matched (2-way / 3-way).",
    "What are the top recurring reasons for invoice mismatch, holds, or rework?",
    "How is the payment cycle managed (terms, early/late payments, exceptions)?",
    "Describe the approval matrix and escalation paths for P2P. Any informal workarounds?",
    "How is the delegation of authority (DoA) maintained and updated?",
    "How are P2P process exceptions tracked (e.g., manual holds, off-cycle payments, urgent requests)?",
    "What internal or external audit observations have you had related to P2P in the last 2–3 years?",
    "How different are P2P processes across countries/entities/business units?",
    "Where is P2P process documentation (SOPs, process maps) stored, and how up-to-date is it?",
    "How are non-PO invoices handled and controlled? Any risks?",
    "How are key cycle times (PR-to-PO, GRN, invoice processing, payment) measured and reported today?"
]

TECH_QUESTIONS = [
    "Which ERP and related systems support your P2P process (per entity)?",
    "To what extent is the P2P flow (PR → PO → GRN → Invoice → Payment) executed fully in system vs. outside?",
    "Describe how approval workflows are configured in the system. Any known gaps or pain points?",
    "How are vendor masters maintained across systems? Any duplication or sync issues?",
    "Describe key integrations (ERP ↔ HR, ERP ↔ banking, ERP ↔ other finance/procurement tools). Any failures?",
    "What automation tools (OCR, RPA, workflow tools) exist for P2P today? How effectively are they used?",
    "How automated is invoice capture and data entry vs. manual keying?",
    "Describe master data quality (vendors, items, GLs, cost centers). Where do you see errors?",
    "What standard reports or dashboards are available today for P2P KPIs and cycle times?",
    "Do teams rely more on system reports or offline Excel sheets for tracking and analysis?",
    "How are system incidents (bugs, downtime, performance issues) logged and resolved?",
    "How are user access roles defined for P2P? Any known SoD (segregation of duties) gaps?",
    "Does the ERP/system support robust 2-way and 3-way matching? Any limitations?",
    "How are exceptions (e.g., unmatched invoices, blocked payments) flagged and routed in the system?",
    "Are there any known system constraints that force manual workarounds in P2P? Describe them.",
    "How easily can you extract an audit trail (who approved what, when) from systems for P2P?"
]

DIMENSIONS = {
    "People": PEOPLE_QUESTIONS,
    "Process": PROCESS_QUESTIONS,
    "Technology": TECH_QUESTIONS,
}

# ---------------------------
# AI HELPERS
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
                    "You MUST follow JSON formatting instructions exactly when requested."
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


def ai_pre_score_responses(client_responses: dict) -> dict:
    """
    Given client_responses = {dimension: [{"question":..., "answer":...}, ...]},
    return ai_scores = {dimension: [{"question":..., "answer":..., "ai_score":int, "reason":str}, ...]}
    We call OpenAI separately per dimension to reduce payload and avoid timeouts.
    """
    overall_result = {}

    base_instructions = """
You will receive a structured object containing responses to a P2P (Procure-to-Pay) diagnostic questionnaire
for ONE dimension: either People, Process, or Technology.

For each answer, you must:
1) Assign a maturity score from 1 to 5 (1 = very poor / ad-hoc, 5 = world-class / fully optimized).
2) Provide a short reason (1–2 lines) explaining why you gave that score.

Important:
- Focus on what is actually described in the answer.
- If an answer indicates heavy manual workarounds, poor controls, lack of documentation, or recurring issues → Score 1–2.
- If partially standardized, with some gaps or inconsistency → Score 3.
- If well-documented, measured, mostly stable → Score 4–5 depending on how strong it sounds.
- If the answer is empty or non-informative, default to score 3 and reason "Insufficient detail; assumed mid-level maturity."

You MUST respond ONLY in valid JSON with this exact structure:

[
  {
    "index": <question_index_starting_from_1>,
    "question": "<question text>",
    "answer": "<client answer>",
    "ai_score": <integer 1-5>,
    "reason": "<short reason>"
  }
]

Do not include any other text outside the JSON array.
"""

    for dim in ["People", "Process", "Technology"]:
        items = client_responses.get(dim, [])
        if not items:
            overall_result[dim] = []
            continue

        prompt = base_instructions + "\n\nDimension: " + dim + "\n\nHere is the client response data (as JSON):\n"
        prompt += json.dumps(items, indent=2)

        raw = call_openai_chat(prompt)
        if raw.startswith("ERROR:"):
            return {"error": f"{dim} scoring failed: {raw}"}

        try:
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            overall_result[dim] = parsed
        except Exception as e:
            return {"error": f"{dim} scoring failed: could not parse AI JSON: {e}\nRaw response was:\n{raw}"}

    return overall_result

# ---------------------------
# SESSION STATE HELPERS
# ---------------------------

def init_session_state():
    if "client_responses" not in st.session_state:
        st.session_state["client_responses"] = {
            "People": [],
            "Process": [],
            "Technology": [],
        }
    if "ai_scoring" not in st.session_state:
        st.session_state["ai_scoring"] = None
    if "consultant_scores" not in st.session_state:
        st.session_state["consultant_scores"] = {
            "People": {},
            "Process": {},
            "Technology": {},
        }
    if "qa_document_text" not in st.session_state:
        st.session_state["qa_document_text"] = None

# ---------------------------
# Q&A DOCUMENT HELPER
# ---------------------------

def build_qa_document_text(client_name: str, assessor_name: str, responses: dict) -> str:
    """Create a plain-text (Word-friendly) document with all questions and answers."""
    lines = []
    lines.append("P2P CoE Diagnostic – Q&A Record")
    lines.append("=" * 60)
    lines.append("")
    if client_name:
        lines.append(f"Client / Entity : {client_name}")
    if assessor_name:
        lines.append(f"Assessor       : {assessor_name}")
    lines.append("")
    lines.append("Note: This document captures raw self-assessment inputs for reference and audit trail.")
    lines.append("")

    for dim in ["People", "Process", "Technology"]:
        dim_items = responses.get(dim, [])
        if not dim_items:
            continue
        lines.append("")
        lines.append(f"{dim} Dimension")
        lines.append("-" * (len(dim) + 11))
        lines.append("")
        for item in dim_items:
            idx = item.get("index")
            q = (item.get("question", "") or "").strip()
            a = (item.get("answer", "") or "").strip()
            lines.append(f"Q{idx}. {q}")
            if a:
                lines.append(f"   A: {a}")
            else:
                lines.append("   A: [No response provided]")
            lines.append("")

    return "\n".join(lines)

# ---------------------------
# UI SECTIONS
# ---------------------------

def render_client_assessment():
    st.subheader("Client / Self-Assessment")
    st.caption(
        "This section is designed to be filled by the client or key stakeholders. "
        "No numeric scores are shown here – only descriptive inputs."
    )

    client_name = st.text_input("Client / Entity Name", value="Bain Global")
    assessor_name = st.text_input("Name of person filling this assessment", value="")

    st.markdown("---")

    responses = {"People": [], "Process": [], "Technology": []}

    for dim, questions in DIMENSIONS.items():
        st.markdown(f"### {dim} – Qualitative Inputs")
        for i, q in enumerate(questions, start=1):
            key = f"{dim}_q_{i}"
            answer = st.text_area(
                f"{dim} Q{i}: {q}",
                key=key,
                height=80,
            )
            responses[dim].append({"index": i, "question": q, "answer": answer})
        st.markdown("---")

    # Run AI pre-scoring
    if st.button("Run AI Pre-Scoring (Based on Answers)"):
        st.session_state["client_responses"] = responses
        with st.spinner("Calling AI to pre-score responses..."):
            ai_result = ai_pre_score_responses(responses)
        st.session_state["ai_scoring"] = ai_result
        if "error" in ai_result:
            st.error(ai_result["error"])
        else:
            st.success("AI pre-scoring completed. Go to 'Consultant Review & Dashboard' tab to view and override scores.")

    # Q&A document creation and download
    st.markdown("### Download Q&A Record")
    if st.button("Prepare Q&A Document for Download"):
        st.session_state["client_responses"] = responses
        qa_text = build_qa_document_text(client_name, assessor_name, responses)
        st.session_state["qa_document_text"] = qa_text
        st.success("Q&A document prepared. Use the download button below.")

    qa_text = st.session_state.get("qa_document_text", None)
    if qa_text:
        file_name = (client_name or "Client") + "_P2P_CoE_QA.txt"
        st.download_button(
            label="Download Q&A as Text File",
            data=qa_text.encode("utf-8"),
            file_name=file_name.replace(" ", "_"),
            mime="text/plain",
        )

    return client_name, assessor_name


def render_consultant_review_and_dashboard():
    st.subheader("Consultant Review & Dashboard")
    st.caption(
        "This section is for SSC consultants to review AI-pre-scored maturity, override scores, "
        "and generate the final dashboard and narrative."
    )
    ai_scoring = st.session_state.get("ai_scoring", None)

    if not ai_scoring or "error" in ai_scoring:
        st.warning("No valid AI scoring available. Please complete the Client Assessment and run AI pre-scoring first.")
        return

    # Sidebar weights
    st.sidebar.markdown("### Weight Configuration")
    people_w = st.sidebar.slider("People Weight %", 0, 100, 30, 5)
    process_w = st.sidebar.slider("Process Weight %", 0, 100, 40, 5)
    tech_w = st.sidebar.slider("Technology Weight %", 0, 100, 30, 5)
    total_w = people_w + process_w + tech_w
    if total_w != 100:
        st.sidebar.warning(f"Current total weight = {total_w}%. Consider adjusting to 100%.")

    dim_scores_ai = {}
    dim_scores_final = {}
    export_rows = []  # to collect full question-level data for CSV export

    # For each dimension, show table with AI scores & override options
    for dim in ["People", "Process", "Technology"]:
        st.markdown(f"### {dim} – Question-level Review")
        ai_items = ai_scoring.get(dim, [])
        rows = []

        if not isinstance(ai_items, list):
            st.error(f"AI scoring format for {dim} is invalid.")
            continue

        # Build rows and override widgets
        for item in ai_items:
            idx = item.get("index")
            question = item.get("question", "")
            answer = item.get("answer", "")
            ai_score = int(item.get("ai_score", 3))
            reason = item.get("reason", "")

            override_key = f"{dim}_override_{idx}"
            default_value = ai_score
            final_score = st.selectbox(
                f"{dim} Q{idx} – Final Score (1–5)",
                options=[1, 2, 3, 4, 5],
                index=(default_value - 1) if 1 <= default_value <= 5 else 2,
                key=override_key,
            )

            st.write(f"**Question {idx}:** {question}")
            st.write(f"**Client Answer:** {answer if answer.strip() else '_No answer provided_'}")
            st.write(f"**AI Suggested Score:** {ai_score}  \n**AI Reason:** {reason}")
            st.write(f"**Consultant Final Score:** {final_score}")
            st.markdown("---")

            row = {
                "dimension": dim,
                "index": idx,
                "question": question,
                "answer": answer,
                "ai_score": ai_score,
                "final_score": final_score,
                "reason": reason,
            }
            rows.append(row)
            export_rows.append(row)

        df_dim = pd.DataFrame(rows)
        if not df_dim.empty:
            dim_scores_ai[dim] = df_dim["ai_score"].mean()
            dim_scores_final[dim] = df_dim["final_score"].mean()
        else:
            dim_scores_ai[dim] = 0
            dim_scores_final[dim] = 0

    # Compute weighted score based on final scores
    total_w = people_w + process_w + tech_w
    if total_w == 0:
        people_w_norm = process_w_norm = tech_w_norm = 0
    else:
        people_w_norm = people_w / total_w
        process_w_norm = process_w / total_w
        tech_w_norm = tech_w / total_w

    overall_weighted_final = (
        dim_scores_final["People"] * people_w_norm +
        dim_scores_final["Process"] * process_w_norm +
        dim_scores_final["Technology"] * tech_w_norm
    )

    st.markdown("## Maturity Scores (AI vs Consultant Final)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("People (Final)", f"{dim_scores_final['People']:.2f} / 5", f"AI: {dim_scores_ai['People']:.2f}")
    col2.metric("Process (Final)", f"{dim_scores_final['Process']:.2f} / 5", f"AI: {dim_scores_ai['Process']:.2f}")
    col3.metric("Technology (Final)", f"{dim_scores_final['Technology']:.2f} / 5", f"AI: {dim_scores_ai['Technology']:.2f}")
    col4.metric("Overall (Weighted Final)", f"{overall_weighted_final:.2f} / 5")

    # Radar chart using final scores
    st.markdown("### Radar View (Final Consultant Scores)")
    radar_df = pd.DataFrame({
        "Dimension": ["People", "Process", "Technology"],
        "Final Score": [
            dim_scores_final["People"],
            dim_scores_final["Process"],
            dim_scores_final["Technology"],
        ],
    })
    fig_radar = px.line_polar(
        radar_df,
        r="Final Score",
        theta="Dimension",
        line_close=True,
        range_r=[0, 5]
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Bar chart
    st.markdown("### Bar Chart – Final Dimension Scores")
    fig_bar = px.bar(
        radar_df,
        x="Dimension",
        y="Final Score"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Download option (full question & answer level data)
    st.markdown("### Export Data")
    if export_rows:
        df_export = pd.DataFrame(export_rows)
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Question-Level Data as CSV",
            data=csv_data,
            file_name="p2p_coe_diagnostic_question_level.csv",
            mime="text/csv"
        )
    else:
        st.info("No question-level data available to export.")

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
    st.sidebar.markdown("P2P CoE Diagnostic (v4)")
    st.sidebar.markdown("---")

# ---------------------------
# MAIN
# ---------------------------

def main():
    init_session_state()

    # Top branding
    render_top_branding()

    # Sidebar branding
    render_sidebar_branding()

    tab_client, tab_consultant = st.tabs(["Client Assessment", "Consultant Review & Dashboard"])

    with tab_client:
        render_client_assessment()

    with tab_consultant:
        render_consultant_review_and_dashboard()

    # Footer
    st.markdown(
        """
        <hr>
        <div style='text-align:center;color:gray;font-size:12px;'>
        P2P CoE Diagnostic App • StrategyStack Consulting © 2025<br>
        Prepared for Bain Global under Program RISE
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
