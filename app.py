"""Streamlit UI: preview data -> run reconciliation -> read the report -> ask questions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from recon import MatchConfig, SAMPLE_QUESTIONS, load_batch, load_frames, reconcile  # noqa: E402
from recon.llm import LLMClient  # noqa: E402

DATA_DIR = ROOT / "data"

load_dotenv()
st.set_page_config(page_title="Settlement Reconciliation Agent", layout="wide")
st.title("Settlement Reconciliation + Q&A Agent")
st.caption("Razorpay settlements vs merchant orders: auto-match, explain exceptions, answer questions.")


@st.cache_data(show_spinner=False)
def _read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file, dtype=str, keep_default_na=False)


def _load_inputs(orders_file, settlements_file):
    if orders_file and settlements_file:
        truth = None
        return load_frames(_read_csv(orders_file), _read_csv(settlements_file), truth)
    return load_batch(DATA_DIR / "orders.csv", DATA_DIR / "settlements.csv", DATA_DIR / "ground_truth.csv")


with st.sidebar:
    st.header("Batch")
    orders_file = st.file_uploader("orders.csv", type="csv")
    settlements_file = st.file_uploader("settlements.csv", type="csv")
    st.caption("Leave empty to use the bundled sample batch (with ground truth).")

    st.header("Thresholds")
    amount_tolerance = st.slider("Amount tolerance (±%)", 1, 20, 5) / 100
    date_window = st.slider("Date window (± days)", 1, 10, 3)
    accept_conf = st.slider("Accept confidence", 0.5, 0.99, 0.80, 0.01)

    llm_ready = LLMClient().available
    use_llm = st.toggle("Use the LLM for reasoning + Q&A", value=llm_ready, disabled=not llm_ready)
    st.caption(
        "GROQ_API_KEY detected." if llm_ready
        else "No GROQ_API_KEY - running the deterministic reasoner."
    )
    run = st.button("Run reconciliation", type="primary", use_container_width=True)

if run or "result" not in st.session_state:
    batch = _load_inputs(orders_file, settlements_file)
    with st.spinner("Reconciling batch..."):
        st.session_state.result = reconcile(
            batch,
            config=MatchConfig(
                amount_tolerance=amount_tolerance,
                date_window_days=date_window,
                accept_confidence=accept_conf,
            ),
            use_llm=use_llm,
        )
        st.session_state.qa = st.session_state.result.qa()
        st.session_state.chat = []

result = st.session_state.result
metrics = result.metrics

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Match rate", f"{metrics.match_rate:.1%}")
c2.metric("Matched", metrics.matched, f"{metrics.fuzzy_recovered} via retrieval")
c3.metric("Exceptions", metrics.exceptions, f"{metrics.uncertain} uncertain")
c4.metric(
    "False-match rate",
    "n/a" if metrics.false_match_rate is None else f"{metrics.false_match_rate:.2%}",
    None if metrics.false_matches is None else f"{metrics.false_matches} wrong",
)
c5.metric("Latency", f"{metrics.elapsed_seconds}s", f"{metrics.throughput_per_second}/s")

tabs = st.tabs(["Data", "Reconciliation", "Exceptions", "Q&A"])

with tabs[0]:
    st.subheader("Orders")
    st.dataframe(result.batch.orders, use_container_width=True, height=260)
    st.subheader("Settlements")
    st.dataframe(result.batch.settlements, use_container_width=True, height=260)

with tabs[1]:
    money = result.money
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gross", f"₹{money['gross_amount']:,.2f}")
    m2.metric("Fees", f"₹{money['fees']:,.2f}")
    m3.metric("Tax", f"₹{money['tax']:,.2f}")
    m4.metric("Net payout", f"₹{money['net_payout']:,.2f}")
    status_filter = st.multiselect(
        "Status", sorted(result.matches["match_status"].unique()),
        default=sorted(result.matches["match_status"].unique()),
    )
    view = result.matches[result.matches["match_status"].isin(status_filter)]
    st.dataframe(view, use_container_width=True, height=420)
    st.download_button("Download reconciliation.csv", result.matches.to_csv(index=False),
                       "reconciliation.csv", "text/csv")

with tabs[2]:
    st.subheader("Exception breakdown")
    st.dataframe(result.breakdown, use_container_width=True)
    st.subheader("Every exception, with reasoning")
    exc = result.matches[result.matches["exception_reason"].notna()]
    st.dataframe(
        exc[["settlement_id", "order_id", "exception_reason", "confidence", "reason",
             "gross_amount", "order_amount", "amount_delta", "settlement_lag_days"]],
        use_container_width=True,
        height=320,
    )
    st.subheader("Orders with no settlement")
    st.dataframe(result.outcome.unsettled_orders, use_container_width=True)

with tabs[3]:
    st.caption("Answers are computed over the reconciled records and cite the record IDs used.")
    cols = st.columns(3)
    for i, question in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 3].button(question, use_container_width=True):
            st.session_state.pending_question = question
    typed = st.chat_input("Ask about fees, tax, net payout, refunds, exceptions...")
    question = typed or st.session_state.pop("pending_question", None)

    for entry in st.session_state.get("chat", []):
        with st.chat_message("user"):
            st.write(entry["q"])
        with st.chat_message("assistant"):
            st.write(entry["a"])
            st.caption("cited: " + (", ".join(entry["ids"][:15]) or "none"))

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"), st.spinner("Checking the batch..."):
            answer = st.session_state.qa.ask(question)
            st.write(answer.answer)
            st.caption("cited: " + (", ".join(answer.cited_records[:15]) or "none"))
            with st.expander("Evidence used"):
                st.json(answer.evidence, expanded=False)
        st.session_state.chat.append({"q": question, "a": answer.answer, "ids": answer.cited_records})
