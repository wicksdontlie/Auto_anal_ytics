import streamlit as st
from analyst import (
    load_data,
    suggest_prompts,
    prompt_to_code,
    run_code,
    ask_llm,
    build_dataset_context,
    is_code_request
)
import pandas as pd

st.set_page_config(page_title="Personal AI Data Analyst", layout="wide")
st.title("🧠 Personal AI Data Analyst — Interactive Dashboard")

st.sidebar.header("Settings")
use_llm = st.sidebar.checkbox("Use local LLM (ollama) for custom prompts", value=False)
llm_model = st.sidebar.text_input("LLM model name (ollama)", value="llama3.1")
st.sidebar.markdown("If you don't have `ollama` installed, leave this off and use built-in prompts.")

uploaded = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv","xls","xlsx","json"])
if uploaded is None:
    st.info("Upload a CSV / XLSX / JSON to get started. Suggestions will appear automatically.")
    st.stop()

# Load data
try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.success("File loaded.")
with st.expander("Preview data (first 100 rows)"):
    st.dataframe(df.head(100))

# Generate suggestions
suggestions = suggest_prompts(df)
st.markdown("## Suggested analyses (pick one or write your own)")
col1, col2 = st.columns([3,1])
with col1:
    selected = st.selectbox("Choose a suggested prompt", options=suggestions)
    custom = st.text_area("Or write a custom prompt (leave blank to use the selected suggestion)", height=80)
with col2:
    st.markdown("**Quick actions**")
    if st.button("Show suggestions again"):
        st.write(suggestions)

# Determine final prompt
final_prompt = custom.strip() if custom and custom.strip() else selected
# Determine final prompt
final_prompt = custom.strip() if custom and custom.strip() else selected

# Determine final prompt
final_prompt = custom.strip() if custom and custom.strip() else selected

st.markdown("### Final prompt")
st.write(final_prompt)

# Run button
if st.button("Run analysis"):

    with st.spinner("Running..."):

        # First try deterministic conversion
        code = prompt_to_code(final_prompt, df)

        if code:

            res = run_code(df, code)

        else:

            if use_llm:

                dataset_context = build_dataset_context(df)

                # Decide whether prompt needs code generation
                code_mode = is_code_request(final_prompt)

                # =========================================
                # TEXT ANALYSIS MODE
                # =========================================
                if not code_mode:

                    system = """
You are a senior AI data analyst.

Analyze datasets professionally.

Provide:
- dataset summary
- business insights
- trends
- anomalies
- recommendations

DO NOT generate python code.
Return only analytical explanation.
"""

                    raw = f"""
{system}

DATASET CONTEXT:
{dataset_context}

USER QUESTION:
{final_prompt}
"""

                    llm_out = ask_llm(
                        raw,
                        df,
                        model=llm_model
                    )

                    st.markdown("## AI Analysis")
                    st.write(llm_out)

                    st.stop()

                # =========================================
                # CODE GENERATION MODE
                # =========================================
                else:

                    system = """
You are a senior AI data analyst.

Generate ONLY python code.

RULES:
- Use dataframe named df
- Store output in variable result
- Use matplotlib for charts
- Never use plt.show()
- Handle non-numeric columns safely
- Use pandas efficiently
"""

                    raw = f"""
{system}

DATASET CONTEXT:
{dataset_context}

USER REQUEST:
{final_prompt}
"""

                    llm_out = ask_llm(
                        raw,
                        df,
                        model=llm_model
                    )

                    if llm_out.startswith("[LLM"):

                        st.warning("LLM error")
                        st.write(llm_out)
                        st.stop()

                    import re

                    match = re.search(
                        r"```python(.*?)```",
                        llm_out,
                        re.DOTALL
                    )

                    if match:

                        code = match.group(1).strip()

                        res = run_code(df, code)

                    else:

                        st.error("No python code block returned.")
                        st.write(llm_out)
                        st.stop()

            else:

                st.error(
                    "Custom prompts require local LLM. Enable Ollama in sidebar."
                )

                st.stop()

    # =========================================
    # DISPLAY RESULTS
    # =========================================

    if res["type"] == "text":

        st.markdown("#### Output (text)")
        st.text(res["output"])

    elif res["type"] == "dataframe":

        st.markdown("#### Output (table)")
        st.dataframe(res["df"])

        csv = res["df"].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download result as CSV",
            data=csv,
            file_name="result.csv",
            mime="text/csv"
        )

    elif res["type"] == "image":

        st.markdown("#### Output (chart)")

        st.image(
            res["path"],
            use_container_width=True
        )

    else:

        st.write("Unknown result type", res)
