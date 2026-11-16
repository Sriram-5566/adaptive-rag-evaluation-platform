
# ==========================================
# RAG EVALUATION LAB
# LLM Answer Generation + Evidence Checking
# Version 3 - Grounded Answer Generation
# ==========================================

import json
import re
from ollama import chat


# ==========================================
# STEP 1: NORMALIZE TEXT
# ==========================================

def normalize_text(text):
    """
    Remove extra spaces and line breaks.

    This helps match evidence extracted from
    PDFs even when formatting differs.
    """

    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


# ==========================================
# STEP 2: DETECT ABSTENTION RESPONSES
# ==========================================

def is_abstention(answer):
    """
    Recognize responses indicating that
    the answer is unavailable.

    Returns True when the model abstains.
    """

    answer = normalize_text(answer).lower()

    if not answer:
        return True

    exact_phrases = {
        "unknown",
        "empty string",
        "not specified",
        "not provided",
        "not mentioned",
        "cannot be determined",
        "no information",
        "not found",
        "n/a",
        "none",
        "null"
    }

    if answer in exact_phrases:
        return True

    abstention_patterns = [
        r"\bnot specified in (?:the|this) context\b",
        r"\bnot provided in (?:the|this) context\b",
        r"\bnot mentioned in (?:the|this) context\b",
        r"\bcannot be determined from (?:the|this) context\b",
        r"\bno information (?:is )?provided\b",
        r"\bthe (?:document|context) does not specify\b",
        r"\bthe (?:document|context) does not mention\b",
        r"\bthe (?:document|context) does not provide\b"
    ]

    return any(
        re.search(pattern, answer)
        for pattern in abstention_patterns
    )


# ==========================================
# STEP 3: GENERATE ANSWER USING OLLAMA
# ==========================================

def recover_answer(question, evidence):
    """
    Try to recover an answer when the
    original generator found evidence
    but returned an empty answer.
    """

    prompt = f"""
Answer the QUESTION using ONLY the EVIDENCE.

QUESTION:
{question}

EVIDENCE:
{evidence}

Rules:
1. Extract the exact information requested.
2. If the evidence clearly answers the question,
   return the answer.
3. If the evidence does not answer the question,
   return Unknown.
4. Do not invent information.
5. Do not use outside knowledge.
6. Return ONLY the answer, without explanation.
"""

    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    recovered = response.message.content.strip()

    if is_abstention(recovered):
        return ""

    return recovered


def resolve_yes_no(question, evidence):
    """
    Resolve a yes/no question using verified
    evidence from the retrieved context.

    Returns:
        "Yes", "No", or ""
    """

    
    prompt = f"""
You are a careful yes/no question-answering assistant.

Answer the QUESTION using ONLY the EVIDENCE.

Determine whether the evidence supports:
- Yes: The statement in the question is true.
- No: The statement in the question is false.
- Unknown: The evidence does not establish either.

IMPORTANT REASONING RULES:

1. Read the exact question carefully.

2. Distinguish between similar but different
   concepts, such as scheduled appointments
   and walk-in appointments.

3. Understand ordinary relationships:
   - Saturday and Sunday are weekend days.
   - Monday through Friday are weekdays.
   - "Does not operate on weekends" means
     the service does not operate on Saturday
     or Sunday.

4. Explicit negative statements are valid
   evidence for answering "No".

5. Do not assume that something is prohibited
   merely because it is not mentioned.

6. If the evidence clearly confirms or denies
   the question, do not answer Unknown.

7. Do not use outside facts about the university.

QUESTION:
{question}

EVIDENCE:
{evidence}

Return ONLY one word:
Yes, No, or Unknown.
"""


    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    decision = response.message.content.strip().lower()

    print("\n===== YES/NO VERIFIER =====")
    print("Question:", question)
    print("Evidence:", evidence)
    print("Decision:", repr(decision))
    print("===========================\n")


    if decision == "yes":
        return "Yes"

    if decision == "no":
        return "No"

    return ""


def generate_answer(question, context):
    """
    Generate a complete, context-grounded answer
    using the local Qwen 2.5 3B model.

    Returns:
        {
            "answer": str,
            "evidence": str,
            "evidence_found": bool
        }

    An empty answer means the model abstained.

    evidence_found indicates whether the
    supplied evidence is a verbatim substring
    of the retrieved context.

    It does NOT establish that the evidence
    supports every claim in the answer.
    """

    question = normalize_text(question)

    if not isinstance(context, str):
        context = ""

    # ======================================
    # HANDLE EMPTY CONTEXT
    # ======================================

    if not context.strip():
        return {
            "answer": "",
            "evidence": "",
            "evidence_found": False
        }

    # ======================================
    # BUILD GENERATION PROMPT
    # ======================================

  
    prompt = f"""
You are a document question-answering assistant.

Answer the QUESTION using ONLY the CONTEXT.

RULES:
1. Read the entire question carefully.
2. Answer the exact question being asked.
   Identify the main requirement, condition,
   location, amount, or date requested.
   Do not replace the requested information
   with a related procedure or subsequent step.
   If multiple details are requested,
   include all of them.
3. Include all requested facts that the context
   supports, including locations, fees, dates,
   conditions, and requirements.
4. Use concise, complete sentences.
5. Do not invent information or use outside facts.
6. For yes/no questions, answer Yes or No when
   the context establishes the answer.
7. If the context does not contain enough
   information to answer the question, use
   an empty answer.
8. Copy the exact supporting sentence or
   consecutive sentences into "evidence".
9. The evidence must appear in the context.
10. Return ONLY a JSON object.

CONTEXT:
{context}

QUESTION:
{question}

REQUIRED JSON:
{{
    "answer": "",
    "evidence": ""
}}

Fill both fields when the question can be answered.
If the answer is unavailable, leave both fields empty.
"""


    # ======================================
    # CALL OLLAMA
    # ======================================

    
    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    # DEBUG: Show the raw response from Qwen
    print("\n===== RAW QWEN RESPONSE =====")
    print(repr(response.message.content))
    print("============================\n")

    # ======================================
    # PARSE JSON RESPONSE
    # ======================================

    try:
        result = json.loads(
            response.message.content
        )

        if not isinstance(result, dict):
            result = {}

    except (json.JSONDecodeError, TypeError):
        result = {}

    # ======================================
    # EXTRACT ANSWER AND EVIDENCE
    # ======================================

    answer = result.get("answer", "")
    evidence = result.get("evidence", "")

    if not isinstance(answer, str):
        answer = ""

    if not isinstance(evidence, str):
        evidence = ""

    answer = normalize_text(answer)
    evidence = normalize_text(evidence)


    # Track actual Ollama calls.
    llm_calls = 1

    # Keep the model's extracted evidence
    # even if its answer is empty.
    original_evidence = evidence

    # Check whether the question is yes/no.
    yes_no_question = bool(
        re.match(
            r"^(is|are|does|do|can|must|has|have|was|were|will|should)\b",
            question,
            flags=re.IGNORECASE
        )
    )

    # Only use evidence that appears in
    # the retrieved context.
    evidence_is_valid = (
        bool(original_evidence)
        and original_evidence in normalize_text(context)
    )

    # If the model found evidence but failed
    # to answer, try a focused yes/no call.

    # Verify every yes/no answer using
    # evidence that appears in the context.
    if yes_no_question and evidence_is_valid:

        llm_calls += 1

        verified_answer = resolve_yes_no(
            question,
            original_evidence
        )

        if verified_answer in {"Yes", "No"}:
            answer = verified_answer
            evidence = original_evidence
        else:
            # The evidence does not establish
            # a reliable yes/no answer.
            answer = ""
            evidence = ""


    # Recover missing answers for non-yes/no
    # questions when valid evidence exists.
    if (
        not yes_no_question
        and is_abstention(answer)
        and evidence_is_valid
    ):
        llm_calls += 1

        recovered_answer = recover_answer(
            question,
            original_evidence
        )

        if recovered_answer:
            answer = recovered_answer
            evidence = original_evidence


    # ======================================
    # HANDLE UNKNOWN ANSWERS
    # ======================================

    if is_abstention(answer):
        answer = ""
        evidence = ""

    # ======================================
    # CHECK SUPPORTING EVIDENCE
    # ======================================

    normalized_context = normalize_text(context)
    normalized_evidence = normalize_text(evidence)

    evidence_found = (
        bool(answer)
        and bool(normalized_evidence)
        and normalized_evidence in normalized_context
    )

    # IMPORTANT:
    # Do not automatically erase an answer
    # when evidence_found is False.
    #
    # QACA needs the original answer and
    # evidence to perform evidence repair
    # and adaptive context expansion.

    # ======================================
    # RETURN RESULTS
    # ======================================


    return {
        "answer": answer,
        "evidence": evidence,
        "evidence_found": evidence_found,
        "llm_calls": llm_calls
    }



# ==========================================
# STEP 4: CHECK EVIDENCE COVERAGE
# ==========================================

def check_evidence_coverage(question, answer, evidence):
    """
    Check whether the evidence supports
    EVERY factual claim in the answer.

    Returns:
        {
            "judgment": str,
            "reason": str
        }

    Possible judgments:
        supported
        unsupported
        uncertain
    """

    if not answer or not evidence:
        return {
            "judgment": "unsupported",
            "reason": "Answer or evidence is missing."
        }

    prompt = f"""
You are a strict evidence verification assistant.

Your task is to determine whether the EVIDENCE
supports EVERY factual claim in the ANSWER.

IMPORTANT:
Supporting only SOME claims is NOT sufficient.

RULES:

1. Identify every factual claim in the answer.

2. Check each claim individually against
   the supplied evidence.

3. Return "supported" ONLY if ALL factual
   claims are supported by the evidence.

4. Return "unsupported" if even ONE factual
   claim is missing from the evidence or
   contradicted by the evidence.

5. Return "uncertain" only when the relationship
   between a claim and the evidence is
   genuinely ambiguous.

6. Do not use outside knowledge.

EXAMPLE 1:

Question:
Where can students replace a lost ID card,
and what is the fee?

Answer:
Students can replace a lost ID card at the
Student Services Desk in Building B.
The fee is 15 dollars.

Evidence:
The replacement fee is 15 dollars.

Judgment: unsupported

Reason:
The evidence supports the fee but does not
support the replacement location.

EXAMPLE 2:

Question:
Where can students replace a lost ID card,
and what is the fee?

Answer:
Students can replace a lost ID card at the
Student Services Desk in Building B.
The fee is 15 dollars.

Evidence:
Lost ID cards can be replaced at the Student
Services Desk in Building B.
The replacement fee is 15 dollars.

Judgment: supported

Reason:
Both the location and the fee are supported.

NOW EVALUATE:

QUESTION:
{question}

ANSWER:
{answer}

EVIDENCE:
{evidence}

Return ONLY valid JSON:

{{
    "judgment": "supported or unsupported or uncertain",
    "reason": "Explain which claims are supported or missing"
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
            options={
                "temperature": 0
            }
        )

        result = json.loads(
            response.message.content
        )

        if not isinstance(result, dict):
            raise ValueError(
                "Expected a JSON object"
            )

        judgment = str(
            result.get("judgment", "uncertain")
        ).strip().lower()

        if judgment not in {
            "supported",
            "unsupported",
            "uncertain"
        }:
            judgment = "uncertain"

        return {
            "judgment": judgment,
            "reason": str(
                result.get("reason", "")
            )
        }

    except Exception as error:
        return {
            "judgment": "uncertain",
            "reason": (
                f"Evidence checker error: {error}"
            )
        }
