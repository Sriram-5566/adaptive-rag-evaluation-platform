
# ==========================================
# RAG EVALUATION LAB
# QACA V2.4 - Robust Evidence Repair
# ==========================================

import json
import re
import time

from ollama import chat

from src.generation import (
    generate_answer,
    check_evidence_coverage,
    normalize_text
)


# ==========================================
# 1. SPLIT CONTEXT INTO SENTENCES
# ==========================================

def split_context_sentences(context):
    """
    Split retrieved context into sentences.
    """

    context = normalize_text(context)

    if not context:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        context
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ==========================================
# 2. VERIFY EVIDENCE COVERAGE
# ==========================================

def verify_evidence(question, output, context):
    """
    Check whether the evidence exists
    and supports the generated answer.
    """

    evidence = output.get("evidence", "")

    evidence_found = (
        bool(normalize_text(evidence))
        and normalize_text(evidence)
        in normalize_text(context)
    )

    if not evidence_found:
        return {
            "judgment": "unsupported",
            "reason": "Evidence not found in context."
        }, 0

    result = check_evidence_coverage(
        question,
        output.get("answer", ""),
        evidence
    )

    return result, 1


# ==========================================
# 3. REPAIR EVIDENCE
# ==========================================

def repair_evidence(question, answer, context):
    """
    First ask Qwen to select supporting sentences.

    If Qwen fails, use the current context
    as a fallback candidate for verification.

    Returns:
        evidence
        llm_calls
        diagnostic reason
    """

    sentences = split_context_sentences(context)

    if not sentences:
        return "", 0, "No sentences available"

    numbered_sentences = "\n".join(
        f"{index}. {sentence}"
        for index, sentence in enumerate(
            sentences,
            start=1
        )
    )

    prompt = f"""
You are an evidence selection assistant.

Select all numbered sentences needed to
support the factual claims in the answer.

RULES:
1. Use only sentence numbers from the list.
2. Include every sentence needed.
3. Do not invent facts.
4. Return an empty list if the answer
   cannot be fully supported.
5. Return valid JSON only.

QUESTION:
{question}

ANSWER:
{answer}

SENTENCES:
{numbered_sentences}

Return:
{{"sentence_ids": []}}

Replace the empty list with the correct IDs.
"""

    selected_ids = []
    repair_error = None

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

        print("\n===== RAW EVIDENCE REPAIR RESPONSE =====")
        print(response.message.content)
        print("========================================\n")

        result = json.loads(response.message.content)

        if isinstance(result, dict):
            raw_ids = result.get("sentence_ids", [])

            if isinstance(raw_ids, list):
                for sentence_id in raw_ids:

                    if isinstance(sentence_id, bool):
                        continue

                    if isinstance(sentence_id, str):
                        if not sentence_id.strip().isdigit():
                            continue

                        sentence_id = int(sentence_id)

                    if (
                        isinstance(sentence_id, int)
                        and 1 <= sentence_id <= len(sentences)
                    ):
                        selected_ids.append(sentence_id)

        selected_ids = sorted(set(selected_ids))

    except Exception as error:
        repair_error = str(error)

    # ======================================
    # TRY MODEL-SELECTED SENTENCES
    # ======================================

    if selected_ids:
        selected_sentences = [
            sentences[index - 1]
            for index in selected_ids
        ]

        evidence = " ".join(selected_sentences)

        # The evidence must be a continuous
        # quotation from the retrieved context.
        if normalize_text(evidence) in normalize_text(context):
            return (
                evidence,
                1,
                f"Selected sentences: {selected_ids}"
            )

    # ======================================
    # FALLBACK: FULL CURRENT CONTEXT
    # ======================================

    # This is only a candidate.
    # run_qaca() will verify it before accepting.
    fallback_evidence = normalize_text(context)

    if repair_error:
        reason = (
            "Sentence selection failed: "
            + repair_error
            + ". Trying full-context fallback."
        )
    else:
        reason = (
            "No usable sentence selection. "
            "Trying full-context fallback."
        )

    return fallback_evidence, 1, reason


# ==========================================
# 4. MAIN QACA V2.4 ALGORITHM
# ==========================================

def run_qaca(
    question,
    chunks,
    retriever,
    relevance_threshold=0.30,
    max_chunks=3
):
    """
    QACA V2.4

    1. Retrieve most relevant chunk.
    2. Generate an answer.
    3. Verify evidence coverage.
    4. Repair incomplete evidence.
    5. Verify repaired evidence.
    6. Expand to neighboring chunks if needed.
    7. Track LLM calls and runtime.
    """

    start_time = time.perf_counter()

    if max_chunks < 1:
        raise ValueError(
            "max_chunks must be at least 1"
        )

    # ======================================
    # HANDLE EMPTY CHUNKS
    # ======================================

    if not chunks:
        return {
            "answer": "",
            "evidence": "",
            "evidence_found": False,
            "selected_chunks": [],
            "chunk_count": 0,
            "context_words": 0,
            "llm_calls": 0,
            "generation_calls": 0,
            "verification_calls": 0,
            "repair_calls": 0,
            "total_llm_calls": 0,
            "runtime": 0.0,
            "steps": [],
            "context": "",
            "stop_reason": "No chunks available",
            "coverage_judgment": "unsupported",
            "coverage_reason": "No context available"
        }

    # ======================================
    # INITIAL RETRIEVAL
    # ======================================

    ranked_results = retriever.retrieve(
        question,
        chunks,
        top_k=len(chunks)
    )

    scores = {
        item["chunk_id"] - 1: item["score"]
        for item in ranked_results
    }

    best_index = ranked_results[0]["chunk_id"] - 1

    selected_indices = [best_index]

    steps = [
        {
            "action": "Initial retrieval",
            "chunk_id": best_index + 1,
            "similarity": scores[best_index]
        }
    ]

    generation_calls = 0
    verification_calls = 0
    repair_calls = 0

    output = {
        "answer": "",
        "evidence": "",
        "evidence_found": False
    }

    coverage = {
        "judgment": "unsupported",
        "reason": "Not checked"
    }

    # ======================================
    # EVALUATE CURRENT CONTEXT
    # ======================================

    def evaluate_current_context():

        nonlocal output
        nonlocal coverage
        nonlocal generation_calls
        nonlocal verification_calls
        nonlocal repair_calls

        ordered_indices = sorted(selected_indices)

        current_context = "\n\n".join(
            chunks[index]
            for index in ordered_indices
        )

        # ----------------------------------
        # GENERATE ANSWER
        # ----------------------------------

        output = generate_answer(
            question,
            current_context
        )


        generation_calls += output.get("llm_calls", 1)

        steps.append({
            "action": "Generated answer",
            "answer": output.get("answer", ""),
            "evidence": output.get("evidence", ""),
            "evidence_found": output.get(
                "evidence_found",
                False
            )
        })

        # ----------------------------------
        # VERIFY ORIGINAL EVIDENCE
        # ----------------------------------

        coverage, calls = verify_evidence(
            question,
            output,
            current_context
        )

        verification_calls += calls

        steps.append({
            "action": "Evidence coverage check",
            "judgment": coverage["judgment"],
            "reason": coverage["reason"]
        })

        if coverage["judgment"] == "supported":
            return True

        if not output.get("answer", "").strip():
            return False

        # ----------------------------------
        # REPAIR EVIDENCE
        # ----------------------------------

        repaired, calls, repair_reason = repair_evidence(
            question,
            output["answer"],
            current_context
        )

        repair_calls += calls

        steps.append({
            "action": "Evidence repair attempted",
            "repaired_evidence": repaired,
            "reason": repair_reason
        })

        if not repaired:
            return False

        # ----------------------------------
        # CHECK REPAIRED QUOTATION
        # ----------------------------------

        normalized_context = normalize_text(
            current_context
        )

        normalized_evidence = normalize_text(
            repaired
        )

        evidence_found = (
            bool(normalized_evidence)
            and normalized_evidence in normalized_context
        )

        if not evidence_found:
            steps.append({
                "action": "Repair rejected",
                "reason": (
                    "Repaired evidence does not "
                    "exist in retrieved context"
                )
            })

            return False

        repaired_output = {
            "answer": output["answer"],
            "evidence": repaired,
            "evidence_found": True
        }

        # ----------------------------------
        # VERIFY REPAIRED EVIDENCE
        # ----------------------------------

        repaired_coverage, calls = verify_evidence(
            question,
            repaired_output,
            current_context
        )

        verification_calls += calls

        steps.append({
            "action": "Repaired evidence verification",
            "judgment": repaired_coverage["judgment"],
            "reason": repaired_coverage["reason"]
        })

        coverage = repaired_coverage

        if coverage["judgment"] == "supported":
            output = repaired_output
            return True

        return False

    # ======================================
    # INITIAL EVALUATION
    # ======================================

    supported = evaluate_current_context()

    stop_reason = None

    if supported:
        stop_reason = (
            "Answer supported by verified evidence"
        )

    else:

        # ==================================
        # FIND IMMEDIATE NEIGHBORS
        # ==================================

        neighbors = [
            index
            for index in [
                best_index - 1,
                best_index + 1
            ]
            if 0 <= index < len(chunks)
        ]

        neighbors.sort(
            key=lambda index: scores[index],
            reverse=True
        )

        # ==================================
        # ADAPTIVE CONTEXT EXPANSION
        # ==================================

        for index in neighbors:

            if len(selected_indices) >= max_chunks:
                stop_reason = "Maximum chunks reached"
                break

            score = scores[index]

            if score < relevance_threshold:
                steps.append({
                    "action": "Skipped: below threshold",
                    "chunk_id": index + 1,
                    "similarity": score
                })

                continue

            selected_indices.append(index)

            steps.append({
                "action": "Added neighboring chunk",
                "chunk_id": index + 1,
                "similarity": score
            })

            supported = evaluate_current_context()

            if supported:
                stop_reason = (
                    "Answer supported after "
                    "context expansion"
                )

                break

    # ======================================
    # FINAL STOP REASON
    # ======================================

    if stop_reason is None:

        if len(selected_indices) >= max_chunks:
            stop_reason = "Maximum chunks reached"

        else:
            stop_reason = (
                "No further eligible adjacent chunks"
            )

    # ======================================
    # FINAL CONTEXT
    # ======================================

    final_indices = sorted(selected_indices)

    final_context = "\n\n".join(
        chunks[index]
        for index in final_indices
    )

    # ======================================
    # METRICS
    # ======================================

    total_calls = (
        generation_calls
        + verification_calls
        + repair_calls
    )

    runtime = time.perf_counter() - start_time

    steps.append({
        "action": "Stopped",
        "reason": stop_reason
    })

    # ======================================
    # RETURN RESULTS
    # ======================================

    return {
        "answer": output.get("answer", ""),
        "evidence": output.get("evidence", ""),
        "evidence_found": output.get(
            "evidence_found",
            False
        ),

        "selected_chunks": [
            index + 1
            for index in final_indices
        ],

        "chunk_count": len(final_indices),

        "context_words": len(
            final_context.split()
        ),

        # Kept for compatibility with app.py.
        "llm_calls": generation_calls,

        "generation_calls": generation_calls,
        "verification_calls": verification_calls,
        "repair_calls": repair_calls,
        "total_llm_calls": total_calls,

        "runtime": runtime,
        "steps": steps,
        "context": final_context,
        "stop_reason": stop_reason,

        "coverage_judgment": coverage["judgment"],
        "coverage_reason": coverage["reason"]
    }
