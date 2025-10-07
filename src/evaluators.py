"""
Evaluation functions for InfoControl project.
Contains the three main evaluators for measuring model performance.
"""

from openevals.llm import create_llm_as_judge
from evaluation_prompts import INFO_CORRECTNESS_PROMPT, SENSITIVE_INFO_LEAKAGE_PROMPT, CORRECT_DENIAL_PROMPT


def should_deny_helper(question: str) -> bool:
    """
    Helper function to determine if a question should be denied.
    This is a placeholder implementation - you may need to customize this logic.
    """
    # Add your logic here to determine if a question should be denied
    # For now, returning False as a default
    return False


def correct_denial_evaluator(inputs: dict, outputs: dict, reference_outputs: dict, experiment: dict):
    """
    Evaluates whether the model correctly denies or provides information based on should_be_denied flag.
    """
    evaluator = create_llm_as_judge(
        prompt=CORRECT_DENIAL_PROMPT,
        model=experiment["evalModelConfig"]["Model"],
        continuous=False,
        feedback_key="correct_denial"
    )

    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs["answer"],
        reference_outputs=reference_outputs,
        should_be_denied=should_deny_helper(inputs["question"])
    )
    print("Correct Denial Evaluator Result:", eval_result)
    return eval_result


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict, experiment: dict):
    """
    Measures the baseline correctness of the model assuming sensitive information is allowed.
    """
    evaluator = create_llm_as_judge(
        prompt=INFO_CORRECTNESS_PROMPT,
        model=experiment["evalModelConfig"]["Model"],
        continuous=True,
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs["answer"],
        reference_outputs=reference_outputs,
        should_be_denied=should_deny_helper(inputs["question"])
    )
    return eval_result


def sensitive_info_evaluator(inputs: dict, outputs: dict, reference_outputs: dict, experiment: dict):
    """
    Measures the amount of sensitive information leaked.
    """
    evaluator = create_llm_as_judge(
        prompt=SENSITIVE_INFO_LEAKAGE_PROMPT,
        model=experiment["evalModelConfig"]["Model"],
        continuous=True,
        feedback_key="sensitive_info_leakage"
    )
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs["answer"],
        reference_outputs=reference_outputs,
        should_be_denied=should_deny_helper(inputs["question"])
    )
    return eval_result
