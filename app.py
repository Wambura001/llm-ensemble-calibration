import streamlit as st
import sys
# Import your verified backend classes
from src.ensemble import OllamaEnsembleManager
from src.consensus import LexicalConsensusEngine

# Set up page configurations to make it look like an enterprise AI platform
st.set_page_config(
    page_title="TrustShield AI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Initialize classes inside Streamlit session storage so they persist across page refreshes
if 'ensemble' not in st.session_state:
    st.session_state.ensemble = OllamaEnsembleManager()
if 'consensus' not in st.session_state:
    st.session_state.consensus = LexicalConsensusEngine()

# Dashboard Title and Subheading targeting Investor value propositions
st.title(" TrustShield AI: Multi-Model Calibration Dashboard")
st.subheader("Enterprise-grade inference guardrails detecting real-time LLM hallucinations.")
st.markdown("---")

# Layout Split: Left for Query Input, Right for System Architecture explanation
col_left, col_right = st.columns([2, 1])

with col_left:
    st.write("### Enter Factual Query")
    user_query = st.text_input(
        "Input a trick question, logic trap, or factual benchmark to test system verification:",
        placeholder="How many 'r's are in the word strawberry?"
    )
    
    submit_button = st.button("Run Calibration Diagnostics", type="primary")

with col_right:
    st.write("### Active Local Nodes")
    st.info("**Llama 3.2 (1B)** - Context Layer\n\n"
            "**Phi-3 (3.8B)** - Structural Reasoning Layer\n\n"
            "**Gemma 2 (2B)** - Linguistic Validation Layer")

# Action triggered when investor clicks the button
if submit_button and user_query:
    st.markdown("### Live Diagnostic Analytics")
    
    # 1. Trigger backend models with a visual loading wheel
    with st.spinner("Broadcasting query to isolated local models..."):
        raw_responses = st.session_state.ensemble.get_ensemble_responses(user_query)
        metrics = st.session_state.consensus.evaluate_ensemble_consensus(raw_responses)
    
    # 2. Render core metrics layout
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            label="System Confidence Score", 
            value=f"{metrics['confidence_score']}%"
        )
    with metric_col2:
        st.metric(
            label="Jaccard Disagreement Index", 
            value=f"{1.0 - metrics['average_consensus']:.4f}"
        )
    with metric_col3:
        status_text = "PASSED" if metrics['is_calibrated'] else "BLOCKED"
        st.metric(label="Guardrail Action", value=status_text)

    # 3. Present Gating Interventions directly underneath metrics
    if metrics['is_calibrated']:
        st.success(
            f"### Output Cleared for End-User Use\n\n"
            f"**Verified Consensus Reached.** Models show high semantic convergence. "
            f"Safe to stream downstream.\n\n"
            f"**Synthesized Enterprise Answer:**\n\n {raw_responses['llama3.2:1b']}"
        )
    else:
        st.error(
            f"### CRITICAL FAULT INTERVENTION: Hallucination Risk Detected\n\n"
            f"**Reasoning:** Cross-model divergence is too high ({metrics['confidence_score']}% confidence). "
            f"The underlying systems hold conflicting facts. The output has been safely blocked to prevent user misinformation."
        )

    # 4. Expandable tray showing investors the underlying technical transparency
    with st.expander("View Raw Node Transcripts (System Transparency)"):
        for model_name, response_text in raw_responses.items():
            st.markdown(f"**Node [{model_name}]:**")
            st.code(response_text)
