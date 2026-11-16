
# ==========================================
# RAG EVALUATION LAB
# Automated Evaluation Engine
# ==========================================

import json
import re
import time

import pandas as pd

from ollama import chat
from sklearn.metrics.pairwise import cosine_similarity

from src.generation import generate_answer
from src.qaca import run_qaca


# ==========================================
# 1. NORMALIZE ANSWERS
# ==========================================

def normalize_answer(text):
    """
    Normalize text for exact-match comparison.
    """

    if text is None or pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


# ==========================================
# 2. EXACT-MATCH CORRECTNESS
# ==========================================

def check_correctness(predicted, expected, answerable):
    """
    Strict answer comparison.

    Correct paraphrases may fail this check.
    """

    predicted = normalize_answer(predicted)
    expected = normalize_answer(expected)

    
    if not answerable:
        return predicted in {
            "",
            "empty string",
            "unknown",
            "not found",
            "not specified",
            "not provided",
            "n a"
        }


    return predicted == expected


# ==========================================
# 3. SEMANTIC ANSWER SIMILARITY
# ==========================================

def calculate_answer_similarity(
    predicted,
    expected,
    embedding_model
):
    """
    Calculate semantic similarity.

    Similarity is NOT proof of correctness.
    """

    predicted = str(predicted).strip()
    expected = str(expected).strip()

    if not predicted or not expected:
        return 0.0

    embeddings = embedding_model.encode(
        [predicted, expected]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)


# ==========================================
# 4. FACTUAL CORRECTNESS CHECKER
# ==========================================

def check_factual_correctness(
    predicted,
    expected,
    answerable=True,
    question=None
):
    """
    Evaluate factual equivalence.

    Returns:
        judgment: correct / incorrect / uncertain
        reason: explanation

    This is an LLM-based judgment, not
    independently verified ground truth.
    """

    predicted = "" if predicted is None else str(predicted).strip()
    expected = "" if expected is None else str(expected).strip()

    
    abstentions = {
        "",
        "empty string",
        "unknown",
        "not found",
        "not specified",
        "not provided",
        "n/a"
    }


    # Handle unanswerable questions.
    if not answerable:

        if predicted.lower() in abstentions:
            return {
                "judgment": "correct",
                "reason": "The system appropriately abstained."
            }

        return {
            "judgment": "incorrect",
            "reason": (
                "The question is unanswerable, "
                "but the system supplied an answer."
            )
        }

    # Handle missing answers.
    if predicted.lower() in abstentions:
        return {
            "judgment": "incorrect",
            "reason": (
                "The system did not answer "
                "an answerable question."
            )
        }

    # Exact normalized match.
    if normalize_answer(predicted) == normalize_answer(expected):
        return {
            "judgment": "correct",
            "reason": "Exact normalized answer match."
        }

    # Ask the local LLM to compare non-identical answers.
    
    prompt = f"""
You are a factual answer correctness evaluator.

Evaluate whether the GENERATED ANSWER correctly
answers the QUESTION using the REFERENCE ANSWER.

The reference answer may be shorter than the
generated answer. Do not require identical wording
or identical length.

RULES:

1. Return "correct" when the generated answer
   provides the factual information requested
   by the question.

2. Accept paraphrases and complete sentences
   that express the same required facts.

3. Do not mark an answer incorrect merely because
   it includes additional information.

4. Return "incorrect" if the generated answer:
   - Gives a wrong factual value.
   - Contradicts the reference answer.
   - Omits information necessary to answer
     the question.
   - Fails to answer the question.

5. Return "uncertain" if additional claims make
   the answer's correctness impossible to determine
   from the question and reference alone.

6. Do not assume additional information is true
   simply because it sounds plausible.

7. Do not use external knowledge.

IMPORTANT:
This evaluation measures answer correctness.
Document grounding will be evaluated separately.

QUESTION:
{question or "Not provided"}

REFERENCE ANSWER:
{expected}

GENERATED ANSWER:
{predicted}

Return ONLY valid JSON:

{{
    "judgment": "correct or incorrect or uncertain",
    "reason": "brief explanation"
}}
"""


    try:

        response = chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={"temperature": 0}
        )

        result = json.loads(response.message.content)

        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")

        judgment = str(
            result.get("judgment", "uncertain")
        ).strip().lower()

        if judgment not in {
            "correct",
            "incorrect",
            "uncertain"
        }:
            judgment = "uncertain"

        return {
            "judgment": judgment,
            "reason": str(result.get("reason", ""))
        }

    except Exception as error:

        return {
            "judgment": "uncertain",
            "reason": f"Evaluator error: {error}"
        }


# ==========================================
# 5. FIXED TOP-K RETRIEVAL
# ==========================================


def run_fixed_retrieval(
    question,
    chunks,
    retriever,
    top_k
):
    """
    Retrieve a fixed number of chunks,
    then generate an answer.

    Count all Ollama calls used during
    answer generation, including the
    focused yes/no verification call.
    """

    start_time = time.perf_counter()

    retrieved = retriever.retrieve(
        question,
        chunks,
        top_k=top_k
    )

    context = "\n\n".join(
        item["text"]
        for item in retrieved
    )

    output = generate_answer(question, context)

    runtime = time.perf_counter() - start_time

    generation_calls = output.get("llm_calls", 1)

    return {
        "answer": output["answer"],
        "evidence": output["evidence"],
        "evidence_found": output["evidence_found"],
        "chunk_count": len(retrieved),
        "context_words": len(context.split()),
        "llm_calls": generation_calls,
        "total_llm_calls": generation_calls,
        "generation_calls": generation_calls,
        "verification_calls": 0,
        "repair_calls": 0,
        "runtime": runtime
    }



# ==========================================
# 6. DATASET EVALUATION
# ==========================================

def run_evaluation(
    dataset,
    chunks,
    retriever,
    relevance_threshold=0.30
):
    """
    Evaluate Top-1, Top-3 and QACA V2
    on every question.
    """

    if not chunks:
        raise ValueError("No document chunks available")

    results = []

    methods = [
        "Top-1",
        "Top-3",
        "QACA V2"
    ]

    for _, row in dataset.iterrows():

        question_id = str(row["question_id"])
        question = str(row["question"])

        expected = row["expected_answer"]

        if pd.isna(expected):
            expected = ""
        else:
            expected = str(expected)

        raw_answerable = row["answerable"]

        if isinstance(raw_answerable, str):
            answerable = (
                raw_answerable.strip().lower() == "true"
            )
        else:
            answerable = bool(raw_answerable)

        for method in methods:

            if method == "Top-1":

                output = run_fixed_retrieval(
                    question,
                    chunks,
                    retriever,
                    top_k=1
                )

            elif method == "Top-3":

                output = run_fixed_retrieval(
                    question,
                    chunks,
                    retriever,
                    top_k=3
                )

            else:

                output = run_qaca(
                    question=question,
                    chunks=chunks,
                    retriever=retriever,
                    relevance_threshold=relevance_threshold,
                    max_chunks=3
                )

            # Strict exact-match metric.
            exact_correct = check_correctness(
                output["answer"],
                expected,
                answerable
            )

            # LLM factual-equivalence judgment.
            factual_result = check_factual_correctness(
                predicted=output["answer"],
                expected=expected,
                answerable=answerable,
                question=question
            )

            # Semantic similarity is diagnostic only.
            # Reuse the retriever's existing model.
            if (
                answerable
                and output["answer"]
                and expected
            ):

                similarity = calculate_answer_similarity(
                    output["answer"],
                    expected,
                    retriever.model
                )

            else:
                similarity = None

            results.append({
                "question_id": question_id,
                "question": question,
                "method": method,
                "expected_answer": expected,
                "generated_answer": (
                    output["answer"] or "Unknown"
                ),
                "answerable": answerable,
                "exact_correct": exact_correct,
                "correct": exact_correct,
                "factual_judgment": (
                    factual_result["judgment"]
                ),
                "factual_reason": (
                    factual_result["reason"]
                ),
                "semantic_similarity": similarity,
                "evidence_found": output["evidence_found"],
                "chunks_used": output["chunk_count"],
                "context_words": output["context_words"],

                "llm_calls": output.get(
                    "total_llm_calls",
                    output["llm_calls"]
                ),
                "generation_calls": output.get(
                    "generation_calls",
                    output["llm_calls"]
                ),
                "verification_calls": output.get(
                    "verification_calls", 0
                ),
                "repair_calls": output.get(
                    "repair_calls", 0
                ),
                "runtime_seconds": round(
                    output["runtime"], 3
                )
            })

    return pd.DataFrame(results)


# ==========================================
# 7. SUMMARIZE RESULTS
# ==========================================

def summarize_results(results):
    """
    Summarize exact-match and factual judgments
    separately.

    Uncertain judgments are not automatically
    counted as correct.
    """

    
    summary = results.groupby("method").agg(
        exact_accuracy=("exact_correct", "mean"),
        avg_chunks=("chunks_used", "mean"),
        avg_context_words=("context_words", "mean"),
        avg_llm_calls=("llm_calls", "mean"),
        avg_generation_calls=("generation_calls", "mean"),
        avg_verification_calls=("verification_calls", "mean"),
        avg_repair_calls=("repair_calls", "mean"),
        avg_runtime=("runtime_seconds", "mean")
    ).reset_index()


    summary["exact_accuracy"] = (
        summary["exact_accuracy"] * 100
    ).round(2)

    factual_counts = (
        results.groupby("method")["factual_judgment"]
        .value_counts()
        .unstack(fill_value=0)
    )

    for label in ["correct", "incorrect", "uncertain"]:
        if label not in factual_counts.columns:
            factual_counts[label] = 0

    factual_counts = factual_counts[
        ["correct", "incorrect", "uncertain"]
    ]

    factual_counts.columns = [
        "factual_correct",
        "factual_incorrect",
        "factual_uncertain"
    ]

    factual_counts["factual_accuracy_percent"] = (
        factual_counts["factual_correct"]
        / factual_counts.sum(axis=1)
        * 100
    ).round(2)

    summary = summary.merge(
        factual_counts.reset_index(),
        on="method",
        how="left"
    )

    # Keep "accuracy" for compatibility with
    # the existing Streamlit accuracy chart.
    summary["accuracy"] = summary["exact_accuracy"]

    return summary.round(3)
