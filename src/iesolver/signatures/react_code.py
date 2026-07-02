"""Phase 4B.4-5 — ReAct code generator + executor signature."""

from __future__ import annotations

import dspy


class ReActCodeSignature(dspy.Signature):
    """
    You are an autonomous Industrial Engineering AI Programmer.
    Your objective is to write, execute, and debug Python code to solve the user's problem.

    You have access to a Python execution tool.
    You MUST adhere strictly to the target library, the library-specific constraints,
    and the requested output specification.

    If the tool returns an error, you must analyze the traceback, fix the code,
    and use the tool again. Keep iterating until the code executes successfully
    and outputs the expected metrics.
    """

    essential_prompt = dspy.InputField(desc="The core analytical problem to solve.")
    target_algorithm = dspy.InputField(desc="The required algorithm to implement.")
    target_library = dspy.InputField(desc="The library you must use (e.g., pulp, sklearn).")
    library_specific_constraints = dspy.InputField(desc="Strict constraints to follow.")
    code_output_spec = dspy.InputField(desc="The exact format of the final printed output.")

    # ReAct döngüsü tamamlandığında modelin dışarı vereceği nihai değişkenler
    final_working_code = dspy.OutputField(desc="The final, error-free Python code snippet.")
    execution_result = dspy.OutputField(desc="The exact printed output from the successful execution.")
