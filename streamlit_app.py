# import streamlit as st
# from app.agent_layer import agentic_cpt_suggestion, agentic_cpt_reverse_lookup
# from app.updater import CPTUpdater
# import json
# import re
# import os
#
# st.set_page_config(
#     page_title="CPT Agentic Assistant",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )
#
# st.title("CPT Agentic Assistant")
# st.markdown(
#     "Enter a doctor's note to get CPT code suggestions with confidence scores, "
#     "or enter a CPT code to see natural language variants. "
#     "You can also add new CPT codes or variants."
# )
#
# # -------------------
# # Sidebar: choose mode
# # -------------------
# mode = st.sidebar.radio("Select Mode", [
#     "Doctor's Note → CPT",
#     "CPT → NL Variants",
#     "Add New CPT / Variant"
# ])
#
# # Initialize updater
# updater = CPTUpdater()
#
# # -------------------
# # Mode 1: Doctor's Note → CPT
# # -------------------
# if mode == "Doctor's Note → CPT":
#     note_input = st.text_area("Enter Doctor's Note:", height=150)
#     top_k = st.slider("Number of candidates to retrieve from FAISS:", 1, 10, 5)
#
#     if st.button("Get CPT Suggestion"):
#         if note_input.strip() == "":
#             st.warning("Please enter a doctor's note.")
#         else:
#             with st.spinner("Running agentic RAG pipeline..."):
#                 result = agentic_cpt_suggestion(note_input, top_k=top_k)
#
#             # Parse raw_output JSON
#             parsed_output = {}
#             raw_output = result.get("raw_output")
#             if raw_output:
#                 cleaned = re.sub(r"^```json|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
#                 try:
#                     parsed_output = json.loads(cleaned)
#                 except Exception as e:
#                     parsed_output = {"raw_output_parse_error": str(e)}
#
#             # Merge top-level fields
#             cpt_code = parsed_output.get("CPT_Code", "N/A")
#             description = parsed_output.get("Description", "N/A")
#             reasoning = parsed_output.get("Reasoning", "N/A")
#             confidence = result.get("Confidence", 0)
#             next_action = result.get("Next_Action", "N/A")
#             error = result.get("error")
#
#             # Display
#             st.subheader("CPT Suggestion")
#             st.markdown(f"**CPT Code:** {cpt_code}")
#             st.markdown(f"**Description:** {description}")
#
#             with st.expander("Reasoning"):
#                 st.write(reasoning)
#
#             conf_color = "green" if confidence >= 0.7 else "orange"
#             st.markdown(
#                 f"**Confidence:** <span style='color:{conf_color}'>{confidence}</span>",
#                 unsafe_allow_html=True
#             )
#             st.markdown(f"**Next Action:** {next_action}")
#
#             if raw_output:
#                 with st.expander("Raw LLM Output"):
#                     st.code(raw_output, language="json")
#             if error:
#                 with st.expander("Error / Parsing Info"):
#                     st.error(error)
#
# # -------------------
# # Mode 2: CPT → NL Variants
# # -------------------
# elif mode == "CPT → NL Variants":
#     cpt_code_input = st.text_input("Enter CPT Code:")
#     if st.button("Get NL Variants"):
#         if cpt_code_input.strip() == "":
#             st.warning("Please enter a CPT code.")
#         else:
#             variants_result = agentic_cpt_reverse_lookup(cpt_code_input.strip())
#             variants = variants_result.get("NL_Variants", [])
#             confidence = variants_result.get("Confidence", 0)
#             next_action = variants_result.get("Next_Action", "N/A")
#
#             if not variants:
#                 st.info(f"No NL variants found for CPT {cpt_code_input}")
#             else:
#                 st.subheader(f"Natural Language Variants for CPT {cpt_code_input}")
#                 for idx, v in enumerate(variants, start=1):
#                     st.write(f"{idx}. {v}")
#
#             conf_color = "green" if confidence >= 0.7 else "orange"
#             st.subheader("Summary")
#             st.markdown(
#                 f"**Confidence:** <span style='color:{conf_color}'>{confidence}</span>",
#                 unsafe_allow_html=True
#             )
#             st.markdown(f"**Next Action:** {next_action}")
#
# # -------------------
# # Mode 3: Add New CPT / Variant
# # -------------------
# else:
#     st.subheader("Add New CPT Code or NL Variant")
#     st.markdown(
#         "You can add a completely new CPT code with its NL variants, "
#         "or add new NL variants to an existing CPT code."
#     )
#
#     cpt_code_input = st.text_input("CPT Code (existing or new)")
#     formal_description_input = st.text_input("Formal Description (required for new CPT code)")
#     nl_variants_input = st.text_area(
#         "Natural Language Variants (comma-separated)",
#         height=100
#     )
#
#     add_as_new = st.checkbox("Add as New CPT Code? (uncheck to add variants to existing CPT code)")
#
#     if st.button("Submit"):
#         if not cpt_code_input.strip() or not nl_variants_input.strip():
#             st.warning("CPT code and at least one NL variant are required.")
#         else:
#             nl_variants_list = [v.strip() for v in nl_variants_input.split(",") if v.strip()]
#             try:
#                 if add_as_new:
#                     if not formal_description_input.strip():
#                         st.warning("Formal description is required for new CPT codes.")
#                     else:
#                         new_entry = updater.add_new_cpt(
#                             cpt_code=cpt_code_input.strip(),
#                             formal_description=formal_description_input.strip(),
#                             nl_variants=nl_variants_list
#                         )
#                         st.success(f"New CPT code {cpt_code_input} added successfully!")
#                         st.json(new_entry)
#                 else:
#                     updated_entry = updater.add_variants(
#                         cpt_code=cpt_code_input.strip(),
#                         new_variants=nl_variants_list
#                     )
#                     st.success(f"NL variants for CPT code {cpt_code_input} updated successfully!")
#                     st.json(updated_entry)
#
#             except Exception as e:
#                 st.error(str(e))



import streamlit as st
from app.agent_layer import agentic_cpt_suggestion, agentic_cpt_reverse_lookup
from app.updater import CPTUpdater
import json
import re

st.set_page_config(
    page_title="CPT Agentic Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("CPT Agentic Assistant")
st.markdown(
    "Enter a doctor's note to get CPT code suggestions with confidence and a verification summary, "
    "enter a CPT code to see natural language variants, or add new CPT codes/variants."
)

mode = st.sidebar.radio("Select Mode", [
    "Doctor's Note → CPT",
    "CPT → NL Variants",
    "Add New CPT / Variant"
])

updater = CPTUpdater()

# ---------- helpers ----------

def _parse_raw_json_block(raw_output: str):
    if not raw_output:
        return {}
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {}

def _badge(label: str, color: str):
    st.markdown(
        f"<span style='display:inline-block;padding:4px 10px;border-radius:8px;"
        f"background:{color};color:white;font-weight:600'>{label}</span>",
        unsafe_allow_html=True
    )

# ---------- Mode 1 ----------

if mode == "Doctor's Note → CPT":
    note_input = st.text_area("Enter Doctor's Note:", height=150)
    top_k = st.slider("Number of candidates to retrieve from FAISS:", 1, 10, 5)

    if st.button("Get CPT Suggestion"):
        if not note_input.strip():
            st.warning("Please enter a doctor's note.")
        else:
            with st.spinner("Running agentic RAG pipeline (with self-critique)..."):
                result = agentic_cpt_suggestion(note_input, top_k=top_k)

            # Parse initial model JSON (if present)
            parsed = _parse_raw_json_block(result.get("raw_output"))

            cpt_code = result.get("CPT_Code") or parsed.get("CPT_Code", "N/A")
            description = result.get("Description") or parsed.get("Description", "N/A")
            reasoning = result.get("Reasoning") or parsed.get("Reasoning", "N/A")
            confidence = result.get("Confidence", 0.0)
            next_action = result.get("Next_Action", "N/A")

            st.subheader("CPT Suggestion")
            st.markdown(f"**CPT Code:** {cpt_code}")
            st.markdown(f"**Description:** {description}")

            with st.expander("Reasoning"):
                st.write(reasoning or "—")

            # Confidence badge
            conf_color = "#16a34a" if confidence >= 0.7 else ("#f59e0b" if confidence >= 0.5 else "#dc2626")
            _badge(f"Confidence: {confidence}", conf_color)
            st.markdown(f"**Next Action:** {next_action}")

            # Verification summary (no chain-of-thought)
            ver = result.get("Verification", {}) or {}
            st.subheader("Verification Summary")
            verdict = (ver.get("verdict") or "warn").upper()
            v_color = {"PASS": "#16a34a", "WARN": "#f59e0b", "FAIL": "#dc2626"}.get(verdict, "#6b7280")
            _badge(f"Verdict: {verdict}", v_color)
            st.markdown(f"**Why:** {ver.get('short_rationale', '—')}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Missing info (if any):**")
                mis = ver.get("missing_info", [])
                if mis:
                    for m in mis:
                        st.write(f"- {m}")
                else:
                    st.write("—")
            with col2:
                st.markdown("**Clarifying questions (if any):**")
                qs = ver.get("clarifying_questions", [])
                if qs:
                    for q in qs:
                        st.write(f"- {q}")
                else:
                    st.write("—")

            # Evidence preview
            ev = result.get("Evidence", []) or []
            if ev:
                with st.expander("Top retrieval evidence (snippets)"):
                    for i, e in enumerate(ev, start=1):
                        st.write(f"{i}. {e.get('text','')}")

            # Debug expanders
            if result.get("raw_output"):
                with st.expander("Raw LLM Output (initial suggestion)"):
                    st.code(result["raw_output"], language="json")

            if result.get("raw_verification_output"):
                with st.expander("Raw Verification Output"):
                    st.code(result["raw_verification_output"], language="json")

            if result.get("error"):
                with st.expander("Error / Parsing Info"):
                    st.error(result["error"])

# ---------- Mode 2 ----------

elif mode == "CPT → NL Variants":
    cpt_code_input = st.text_input("Enter CPT Code:")
    if st.button("Get NL Variants"):
        if not cpt_code_input.strip():
            st.warning("Please enter a CPT code.")
        else:
            res = agentic_cpt_reverse_lookup(cpt_code_input.strip())
            variants = res.get("NL_Variants", [])
            confidence = res.get("Confidence", 0.0)
            next_action = res.get("Next_Action", "N/A")

            if variants:
                st.subheader(f"Natural Language Variants for CPT {cpt_code_input}")
                for i, v in enumerate(variants, start=1):
                    st.write(f"{i}. {v}")
            else:
                st.info(f"No NL variants found for CPT {cpt_code_input}")

            conf_color = "#16a34a" if confidence >= 0.7 else ("#f59e0b" if confidence >= 0.5 else "#dc2626")
            st.subheader("Summary")
            _badge(f"Confidence: {confidence}", conf_color)
            st.markdown(f"**Next Action:** {next_action}")

# ---------- Mode 3 ----------

else:
    st.subheader("Add New CPT Code or NL Variant")
    st.markdown(
        "Add a completely new CPT code with its NL variants, or add new NL variants to an existing CPT code."
    )

    cpt_code_input = st.text_input("CPT Code (existing or new)")
    formal_description_input = st.text_input("Formal Description (required for new CPT code)")
    nl_variants_input = st.text_area("Natural Language Variants (comma-separated)", height=100)
    add_as_new = st.checkbox("Add as New CPT Code? (uncheck to add variants to an existing code)")

    if st.button("Submit"):
        if not cpt_code_input.strip() or not nl_variants_input.strip():
            st.warning("CPT code and at least one NL variant are required.")
        else:
            nl_variants_list = [v.strip() for v in nl_variants_input.split(",") if v.strip()]
            try:
                if add_as_new:
                    if not formal_description_input.strip():
                        st.warning("Formal description is required for new CPT codes.")
                    else:
                        new_entry = updater.add_new_cpt(
                            cpt_code=cpt_code_input.strip(),
                            formal_description=formal_description_input.strip(),
                            nl_variants=nl_variants_list
                        )
                        st.success(f"New CPT code {cpt_code_input} added successfully!")
                        st.json(new_entry)
                else:
                    updated_entry = updater.add_variants(
                        cpt_code=cpt_code_input.strip(),
                        new_variants=nl_variants_list
                    )
                    st.success(f"NL variants for CPT code {cpt_code_input} updated successfully!")
                    st.json(updated_entry)
            except Exception as e:
                st.error(str(e))
