"""The agent loop.

Shape of the loop, for reference (built in Phase 3):

    history = [user message]
    for step in range(MAX_STEPS):
        response = llm.send(history, tools=TOOL_SCHEMAS)
        if no tool calls:      # the model is done talking
            return response.text
        run each tool, append results to history
    stop with a "ran out of steps" message

Phase 1 only needs a single request, so the loop lives in main.py for now
and moves here once there is something to loop over.
"""


def run_agent(task: str) -> str:
    """Run the agent on `task` and return its final reply.

    Not implemented until Phase 3.
    """
    raise NotImplementedError("The agent loop arrives in Phase 3.")
