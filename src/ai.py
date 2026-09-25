import re

_MODEL = None
_TOKENIZER = None


# ============================================================
# MODEL AVAILABILITY
# ============================================================

def llm_available():
    """
    Check whether the local Hugging Face model dependencies
    are available.
    """
    try:
        import transformers
        import torch

        return True

    except Exception:
        return False


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def _numbers(text):
    """
    Extract numerical values from text.

    Examples:
        2,000.50 -> 2000.50
        363.22 -> 363.22
        -4.5 -> -4.5
    """

    values = re.findall(
        r"(?<!\w)-?\d[\d,]*(?:\.\d+)?",
        str(text)
    )

    return {
        v.replace(",", "")
        for v in values
    }


# ============================================================
# NORMALIZE NUMBERS
# ============================================================

def _normalize_number(value):
    """
    Normalize numerical strings so equivalent representations
    can be compared.

    Example:
        363.220 -> 363.22
        2,000 -> 2000
    """

    try:

        number = float(
            str(value).replace(",", "")
        )

        if number.is_integer():

            return str(
                int(number)
            )

        return f"{number:.10f}".rstrip("0").rstrip(".")

    except Exception:

        return str(value)


# ============================================================
# NUMERIC CONSISTENCY CHECK
# ============================================================

def _is_consistent(generated, verified_answer):
    """
    Make sure the AI does not contradict numerical values
    explicitly stated in the verified answer.
    """

    required = _numbers(
        verified_answer
    )

    generated_numbers = _numbers(
        generated
    )

    normalized_required = {
        _normalize_number(x)
        for x in required
    }

    normalized_generated = {
        _normalize_number(x)
        for x in generated_numbers
    }

    return normalized_required.issubset(
        normalized_generated
    )


# ============================================================
# BAD MODEL OUTPUT DETECTION
# ============================================================

def _is_bad_response(text):
    """
    Detect short, meaningless, refusal-like or instruction-like
    outputs from the local model.
    """

    if not text:

        return True

    cleaned = (
        str(text)
        .strip()
        .lower()
    )

    if len(cleaned) < 25:

        return True

    bad_phrases = [

        "do not estimate",
        "don't estimate",
        "cannot estimate",
        "can't estimate",
        "no estimate",
        "not estimate",
        "i cannot",
        "i can't",
        "cannot determine",
        "can't determine",
        "insufficient information",
        "not enough information",
        "i don't know",
        "i cannot answer",
        "unable to answer"

    ]

    for phrase in bad_phrases:

        if phrase in cleaned:

            return True

    return False


# ============================================================
# OPERATION DETECTION
# ============================================================

def _operation_type(details):
    """
    Identify the analytical operation from the engine trace.
    """

    text = str(details).lower()

    if "groupby(" in text:

        if ".count(" in text:

            return "group_count"

        if ".mean(" in text:

            return "group_mean"

        if ".sum(" in text:

            return "group_sum"

        if ".max(" in text:

            return "group_max"

        if ".min(" in text:

            return "group_min"

        return "grouped_analysis"

    if "aggregate.mean" in text:

        return "mean"

    if "aggregate.sum" in text:

        return "sum"

    if "aggregate.max" in text:

        return "max"

    if "aggregate.min" in text:

        return "min"

    if "correlation" in text:

        return "correlation"

    if "ranking" in text:

        return "ranking"

    if "quality.missing" in text:

        return "missing"

    if "quality.duplicates" in text:

        return "duplicates"

    if "describe" in text:

        return "describe"

    return "general"


# ============================================================
# COLUMN EXTRACTION
# ============================================================

def _extract_column(details):
    """
    Try to extract a column name from an analysis trace.
    """

    text = str(details)

    patterns = [

        r"column=([^,\)]+)",

        r"groupby\(([^,\)]+)",

        r"column\s*[:=]\s*([A-Za-z0-9_\- ]+)"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            return match.group(1).strip()

    return None


# ============================================================
# DETERMINISTIC GROUNDED EXPLANATION
# ============================================================

def _fallback(question, answer, details):
    """
    Safe explanation used when the local model is unavailable
    or produces an unreliable response.

    This explanation is generated from the verified analysis
    operation rather than inventing new results.
    """

    operation = _operation_type(
        details
    )

    column = _extract_column(
        details
    )


    # --------------------------------------------------------
    # GROUPED COUNT
    # --------------------------------------------------------

    if operation == "group_count":

        return (
            "NOVA grouped the uploaded dataset by the "
            f"categorical field **{column or 'the selected field'}** "
            "and counted the number of records in each group. "
            "The displayed table contains the verified counts "
            "calculated directly with Pandas."
        )


    # --------------------------------------------------------
    # GROUPED MEAN
    # --------------------------------------------------------

    if operation == "group_mean":

        return (
            "NOVA grouped the dataset and calculated the "
            f"average of **{column or 'the selected numeric field'}** "
            "for each group. The displayed values are calculated "
            "directly from the uploaded data using Pandas."
        )


    # --------------------------------------------------------
    # GROUPED SUM
    # --------------------------------------------------------

    if operation == "group_sum":

        return (
            "NOVA grouped the dataset and calculated the "
            f"total of **{column or 'the selected numeric field'}** "
            "for each group. The result is based directly on "
            "the uploaded dataset."
        )


    # --------------------------------------------------------
    # GROUPED MAX
    # --------------------------------------------------------

    if operation == "group_max":

        return (
            "NOVA grouped the dataset and identified the "
            f"maximum **{column or 'numeric value'}** within "
            "each group. The result was calculated directly "
            "with Pandas."
        )


    # --------------------------------------------------------
    # GROUPED MIN
    # --------------------------------------------------------

    if operation == "group_min":

        return (
            "NOVA grouped the dataset and identified the "
            f"minimum **{column or 'numeric value'}** within "
            "each group. The result was calculated directly "
            "with Pandas."
        )


    # --------------------------------------------------------
    # MEAN
    # --------------------------------------------------------

    if operation == "mean":

        return (
            f"The verified result is {answer}. "
            "NOVA calculated this value directly from the "
            "uploaded dataset using Pandas mean()."
        )


    # --------------------------------------------------------
    # SUM
    # --------------------------------------------------------

    if operation == "sum":

        return (
            f"The verified result is {answer}. "
            "NOVA calculated the total directly from the "
            "uploaded dataset using Pandas sum()."
        )


    # --------------------------------------------------------
    # MAX
    # --------------------------------------------------------

    if operation == "max":

        return (
            f"The verified result is {answer}. "
            "NOVA identified the maximum value directly "
            "from the uploaded dataset using Pandas max()."
        )


    # --------------------------------------------------------
    # MIN
    # --------------------------------------------------------

    if operation == "min":

        return (
            f"The verified result is {answer}. "
            "NOVA identified the minimum value directly "
            "from the uploaded dataset using Pandas min()."
        )


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    if operation == "correlation":

        return (
            f"{answer} "
            "NOVA calculated Pearson correlations between "
            "the available numeric fields using Pandas. "
            "The displayed relationship is based on the "
            "uploaded dataset."
        )


    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    if operation == "ranking":

        return (
            f"{answer} "
            "NOVA ranked the matching records using the "
            "requested numeric field and Pandas ranking "
            "operations."
        )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    if operation == "missing":

        return (
            f"{answer} "
            "NOVA inspected every column using Pandas "
            "missing-value checks."
        )


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    if operation == "duplicates":

        return (
            f"{answer} "
            "NOVA checked the uploaded dataset using "
            "Pandas duplicate-row detection."
        )


    # --------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    if operation == "describe":

        return (
            "NOVA generated a statistical summary of the "
            "available numeric fields using Pandas describe(). "
            "The displayed table contains statistics calculated "
            "directly from the uploaded dataset."
        )


    # --------------------------------------------------------
    # GENERAL FALLBACK
    # --------------------------------------------------------

    return (
        f"The verified result is {answer} "
        "The analysis was calculated directly from the "
        "uploaded dataset using the Pandas analysis engine."
    )


# ============================================================
# LOCAL LLM EXPLANATION
# ============================================================

def explain_with_local_llm(
    question,
    answer,
    details
):
    """
    Generate a grounded natural-language explanation.

    Important:
    The LLM explains the verified Pandas result.
    It does NOT calculate or replace the result.
    """

    fallback = _fallback(
        question,
        answer,
        details
    )


    # --------------------------------------------------------
    # MODEL NOT AVAILABLE
    # --------------------------------------------------------

    if not llm_available():

        return fallback


    try:

        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM
        )


        global _MODEL
        global _TOKENIZER


        # ----------------------------------------------------
        # LOAD MODEL ON FIRST USE
        # ----------------------------------------------------

        if _MODEL is None:

            model_name = (
                "google/flan-t5-small"
            )

            _TOKENIZER = (
                AutoTokenizer.from_pretrained(
                    model_name
                )
            )

            _MODEL = (
                AutoModelForSeq2SeqLM
                .from_pretrained(
                    model_name
                )
            )


        # ----------------------------------------------------
        # OPERATION CONTEXT
        # ----------------------------------------------------

        operation = _operation_type(
            details
        )


        # ----------------------------------------------------
        # GROUNDED PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are explaining the result of a data analysis program.

Write exactly 2 short sentences explaining what the
analysis did and what the verified result means.

USER QUESTION:
{question}

VERIFIED RESULT:
{answer}

ANALYSIS TRACE:
{details}

ANALYSIS TYPE:
{operation}

IMPORTANT:
The verified result was calculated by Pandas and is authoritative.

Do not calculate a new answer.
Do not invent data.
Do not estimate anything.
Do not change numerical values.
Do not contradict the verified result.
Do not say "Do not estimate".
Do not give instructions.
Do not refuse the task.
Explain the operation and the verified result clearly.
"""


        # ----------------------------------------------------
        # TOKENIZE
        # ----------------------------------------------------

        inputs = _TOKENIZER(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )


        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        outputs = _MODEL.generate(

            **inputs,

            max_new_tokens=90,

            num_beams=4,

            early_stopping=True,

            no_repeat_ngram_size=3

        )


        generated = (
            _TOKENIZER.decode(
                outputs[0],
                skip_special_tokens=True
            )
            .strip()
        )


        # ----------------------------------------------------
        # VALIDATE RESPONSE
        # ----------------------------------------------------

        if _is_bad_response(
            generated
        ):

            return fallback


        # ----------------------------------------------------
        # NUMERIC SAFETY CHECK
        # ----------------------------------------------------

        if not _is_consistent(
            generated,
            answer
        ):

            return fallback


        # ----------------------------------------------------
        # RETURN GROUNDED AI EXPLANATION
        # ----------------------------------------------------

        return generated


    except Exception:

        # The verified answer must remain available even
        # when the local model fails.

        return fallback
