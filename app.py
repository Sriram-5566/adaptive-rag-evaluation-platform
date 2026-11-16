from src.evaluation import run_evaluation, summarize_results
from src.qaca import run_qaca
from src.generation import generate_answer
from src.retrieval import SemanticRetriever
import streamlit as st
import pymupdf
import pandas as pd

from src.chunking import (
    fixed_size_chunking,
    sentence_chunking,
    paragraph_chunking,
    overlapping_chunking
)

@st.cache_resource
def load_retriever():
    return SemanticRetriever()

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RAG Evaluation Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 RAG Evaluation Lab")

st.write(
    "A research platform for evaluating document "
    "chunking and retrieval algorithms in RAG systems."
)

st.divider()

# ==========================================
# STEP 1: UPLOAD FILES
# ==========================================

st.header("1. Upload Research Files")

col1, col2, col3 = st.columns(3)

with col1:
    document_file = st.file_uploader(
        "Upload Source Document",
        type=["pdf"],
        key="document"
    )

with col2:
    questions_file = st.file_uploader(
        "Upload Questions",
        type=["csv"],
        key="questions"
    )

with col3:
    answers_file = st.file_uploader(
        "Upload Ground Truth",
        type=["csv"],
        key="answers"
    )

# ==========================================
# STEP 2: EXTRACT PDF TEXT
# ==========================================

if document_file is not None:

    st.divider()
    st.header("2. PDF Text Extraction")

    try:
        pdf = pymupdf.open(
            stream=document_file.getvalue(),
            filetype="pdf"
        )

        pages = []

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text")

            pages.append({
                "page": page_number,
                "text": text
            })

        pdf.close()

        document_text = "\n\n".join(
            page["text"] for page in pages
        )

        if document_text.strip():

            st.success("PDF text extracted successfully!")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric("Total Pages", len(pages))

            with metric2:
                st.metric(
                    "Total Words",
                    len(document_text.split())
                )

            with metric3:
                st.metric(
                    "Total Characters",
                    len(document_text)
                )

            st.subheader("Extracted Text Preview")

            st.text_area(
                "Document Content",
                value=document_text[:10000],
                height=300
            )

            st.caption(
                "Showing up to the first 10,000 characters."
            )

            st.subheader("Page-by-Page Preview")

            selected_page = st.selectbox(
                "Select Page",
                options=list(range(1, len(pages) + 1))
            )

            st.text_area(
                "Page Text",
                value=pages[selected_page - 1]["text"],
                height=250
            )

        else:
            st.error(
                "No selectable text was found. "
                "Please upload a text-based PDF."
            )

    except Exception as error:
        st.error(f"PDF extraction failed: {error}")

# ==========================================
# STEP 3: READ QUESTIONS CSV
# ==========================================

if questions_file is not None:

    st.divider()
    st.header("3. Questions Dataset")

    try:
        questions_df = pd.read_csv(
            questions_file,
            dtype={"question_id": str}
        )

        st.success("Questions CSV uploaded!")

        st.write("Total questions:", len(questions_df))

        st.dataframe(
            questions_df,
            use_container_width=True
        )

    except Exception as error:
        st.error(f"Error reading questions: {error}")

# ==========================================
# STEP 4: READ GROUND TRUTH CSV
# ==========================================

if answers_file is not None:

    st.divider()
    st.header("4. Ground Truth Dataset")

    try:
        answers_df = pd.read_csv(
            answers_file,
            dtype={"question_id": str}
        )

        st.success("Ground truth CSV uploaded!")

        st.write("Total answers:", len(answers_df))

        st.dataframe(
            answers_df,
            use_container_width=True
        )

    except Exception as error:
        st.error(f"Error reading ground truth: {error}")

# ==========================================
# STEP 5: VALIDATE DATASETS
# ==========================================

if questions_file is not None and answers_file is not None:

    st.divider()
    st.header("5. Dataset Validation")

    if "questions_df" in locals() and "answers_df" in locals():

        required_questions = {
            "question_id",
            "question"
        }

        required_answers = {
            "question_id",
            "expected_answer",
            "evidence",
            "answerable"
        }

        if (
            required_questions.issubset(questions_df.columns)
            and required_answers.issubset(answers_df.columns)
        ):

            if (
                questions_df["question_id"].isna().any()
                or answers_df["question_id"].isna().any()
            ):
                st.error("Missing question IDs found.")

            elif (
                questions_df["question_id"].duplicated().any()
                or answers_df["question_id"].duplicated().any()
            ):
                st.error("Duplicate question IDs found.")

            else:

                merged = questions_df.merge(
                    answers_df,
                    on="question_id",
                    how="outer",
                    indicator=True,
                    validate="one_to_one"
                )

                if (merged["_merge"] != "both").any():

                    st.error(
                        "Question IDs do not match "
                        "between the two CSV files."
                    )

                    st.dataframe(
                        merged[
                            merged["_merge"] != "both"
                        ],
                        use_container_width=True
                    )

                else:

                    st.success(
                        "All questions matched "
                        "their ground-truth answers!"
                    )

                    st.metric(
                        "Validated Questions",
                        len(merged)
                    )

                    st.subheader("Combined Evaluation Dataset")

                    st.dataframe(
                        merged.drop(columns=["_merge"]),
                        use_container_width=True
                    )

        else:

            st.error(
                "CSV files are missing required columns."
            )

            st.write(
                "Questions CSV needs: "
                "question_id, question"
            )

            st.write(
                "Ground truth CSV needs: "
                "question_id, expected_answer, "
                "evidence, answerable"
            )

# ==========================================
# STEP 6: EXPERIMENT STATUS
# ==========================================

st.divider()
st.header("6. Experiment Status")

if (
    document_file is not None
    and questions_file is not None
    and answers_file is not None
):

    st.info(
        "Files uploaded. Complete dataset validation "
        "before running experiments."
    )

    st.write(
        "Next: Integrate fixed-size chunking, "
        "semantic retrieval, QACA, and evaluation metrics."
    )

else:

    st.warning(
        "Upload all three files to prepare "
        "your RAG evaluation experiment."
    )


# ==========================================
# STEP 7: CHUNKING EXPERIMENT
# ==========================================

if document_file is not None and "document_text" in locals():

    if document_text.strip():

        st.divider()
        st.header("7. Document Chunking")

        method = st.selectbox(
            "Select Chunking Method",
            [
                "Fixed-size",
                "Sentence-based",
                "Paragraph-based",
                "Overlapping"
            ]
        )

        chunk_size = st.slider(
            "Chunk Size (words)",
            min_value=50,
            max_value=500,
            value=100,
            step=25
        )

        if method == "Fixed-size":

            chunks = fixed_size_chunking(
                document_text,
                chunk_size
            )

        elif method == "Sentence-based":

            sentences_per_chunk = st.slider(
                "Sentences per Chunk",
                min_value=1,
                max_value=10,
                value=3
            )

            chunks = sentence_chunking(
                document_text,
                sentences_per_chunk
            )

        elif method == "Paragraph-based":

            chunks = paragraph_chunking(
                document_text
            )

        elif method == "Overlapping":

            overlap = st.slider(
                "Overlap (words)",
                min_value=0,
                max_value=chunk_size - 1,
                value=min(20, chunk_size - 1)
            )

            chunks = overlapping_chunking(
                document_text,
                chunk_size,
                overlap
            )

        st.subheader("Chunking Statistics")

        total_chunks = len(chunks)

        average_words = (
            sum(len(chunk.split()) for chunk in chunks)
            / total_chunks
            if total_chunks > 0 else 0
        )

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric("Total Chunks", total_chunks)

        with metric2:
            st.metric(
                "Average Words per Chunk",
                round(average_words, 2)
            )

        st.subheader("Chunk Preview")

        for index, chunk in enumerate(chunks):

            with st.expander(
                f"Chunk {index + 1} "
                f"({len(chunk.split())} words)"
            ):
                st.write(chunk)


# ==========================================
# STEP 8: SEMANTIC RETRIEVAL
# ==========================================

if document_file is not None and "chunks" in locals():

    if chunks:

        st.divider()
        st.header("8. Semantic Retrieval")

        question = st.text_input(
            "Ask a question about the PDF",
            value="Where can students purchase parking permits?"
        )

        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=min(10, len(chunks)),
            value=min(3, len(chunks))
        )

        if st.button("Retrieve Relevant Chunks"):

            with st.spinner("Searching document..."):

                retriever = load_retriever()

                results = retriever.retrieve(
                    question,
                    chunks,
                    top_k
                )

            st.subheader("Retrieval Results")

            for result in results:

                st.write(
                    f"**Chunk {result['chunk_id']}** "
                    f"— Similarity: {result['score']:.4f}"
                )

                st.info(result["text"])


# ==========================================
# STEP 9: RAG ANSWER GENERATION
# ==========================================

if document_file is not None and "chunks" in locals():

    if chunks:

        st.divider()
        st.header("9. RAG Answer Generation")

        answer_question = st.text_input(
            "Question for the LLM",
            value="Where can students purchase parking permits?",
            key="answer_question"
        )

        if st.button("Generate RAG Answer"):

            with st.spinner("Retrieving relevant chunks..."):

                retriever = load_retriever()

                results = retriever.retrieve(
                    answer_question,
                    chunks,
                    top_k=3
                )

                context = "\n\n".join(
                    result["text"]
                    for result in results
                )

            with st.spinner("Generating answer with Ollama..."):

                output = generate_answer(
                    answer_question,
                    context
                )

            st.subheader("Generated Answer")

            st.write(
                output["answer"] or "Unknown"
            )

            st.subheader("Supporting Evidence")

            st.write(
                output["evidence"] or "No evidence returned"
            )

            st.write(
                "Exact evidence quote found:",
                output["evidence_found"]
            )

            st.subheader("Retrieved Context")

            st.text_area(
                "Context sent to LLM",
                value=context,
                height=250
            )


# ==========================================
# STEP 10: QACA V2 EXPERIMENT
# ==========================================

if document_file is not None and "chunks" in locals():

    if chunks:

        st.divider()
        st.header("10. QACA V2 Experiment")

        qaca_question = st.text_input(
            "Question for QACA",
            value="Where can students purchase parking permits?",
            key="qaca_question"
        )

        threshold = st.slider(
            "Neighbor relevance threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05
        )

        max_context_chunks = st.slider(
            "Maximum context chunks",
            min_value=1,
            max_value=min(3, len(chunks)),
            value=min(3, len(chunks))
        )

        if st.button("Run QACA V2"):

            with st.spinner("Running QACA..."):

                retriever = load_retriever()

                result = run_qaca(
                    question=qaca_question,
                    chunks=chunks,
                    retriever=retriever,
                    relevance_threshold=threshold,
                    max_chunks=max_context_chunks
                )

            st.subheader("QACA Answer")

            st.write(
                result["answer"] or "Unknown"
            )

            st.subheader("Supporting Evidence")

            st.write(
                result["evidence"] or "No evidence returned"
            )

            st.write(
                "Evidence quote found:",
                result["evidence_found"]
            )

            
            # ======================================
            # QACA V2.4 - EFFICIENCY METRICS
            # ======================================

            st.subheader("Efficiency Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Chunks Used",
                    result["chunk_count"]
                )

            with col2:
                st.metric(
                    "Context Words",
                    result["context_words"]
                )

            with col3:
                st.metric(
                    "Total LLM Calls",
                    result["total_llm_calls"]
                )

            with col4:
                st.metric(
                    "Runtime (seconds)",
                    round(result["runtime"], 3)
                )

            st.subheader("LLM Call Breakdown")

            call1, call2, call3 = st.columns(3)

            with call1:
                st.metric(
                    "Generation Calls",
                    result["generation_calls"]
                )

            with call2:
                st.metric(
                    "Verification Calls",
                    result["verification_calls"]
                )

            with call3:
                st.metric(
                    "Evidence Repair Calls",
                    result["repair_calls"]
                )

            st.caption(
                "Total LLM calls include answer generation, "
                "evidence verification, and evidence repair."
            )



            st.subheader("QACA Decision Steps")

            for step in result["steps"]:
                st.write(step)

            st.write(
                "Stop reason:",
                result.get("stop_reason", "Not available")
            )


            st.subheader("Final Retrieved Context")

            st.text_area(
                "Context selected by QACA",
                value=result["context"],
                height=250
            )


# ==========================================
# STEP 11: AUTOMATED RAG EVALUATION
# ==========================================

if (
    document_file is not None
    and "chunks" in locals()
    and "merged" in locals()
):

    if (
    chunks
    and not merged.empty
    and (merged["_merge"] == "both").all()
):

        st.divider()
        st.header("11. Automated RAG Evaluation")

        st.write(
            "Compare Top-1, Top-3, and QACA V2 "
            "across all validated questions."
        )

        if st.button("Run Full Evaluation"):

            with st.spinner(
                "Running all questions through Ollama..."
            ):

                retriever = load_retriever()

                evaluation_results = run_evaluation(
                    dataset=merged,
                    chunks=chunks,
                    retriever=retriever,
                    relevance_threshold=0.30
                )

                evaluation_summary = summarize_results(
                    evaluation_results
                )

                st.session_state["evaluation_results"] = (
                    evaluation_results
                )

                st.session_state["evaluation_summary"] = (
                    evaluation_summary
                )

        if "evaluation_results" in st.session_state:

            st.subheader("Method Comparison")

            st.dataframe(
                st.session_state["evaluation_summary"],
                use_container_width=True
            )

            st.subheader("Accuracy Comparison")

            chart_data = (
                st.session_state["evaluation_summary"]
                .set_index("method")[["accuracy"]]
            )

            st.bar_chart(chart_data)

            st.subheader("Question-Level Results")

            st.dataframe(
                st.session_state["evaluation_results"],
                use_container_width=True
            )

            csv_data = (
                st.session_state["evaluation_results"]
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                label="Download Evaluation Results CSV",
                data=csv_data,
                file_name="rag_evaluation_results.csv",
                mime="text/csv"
            )


