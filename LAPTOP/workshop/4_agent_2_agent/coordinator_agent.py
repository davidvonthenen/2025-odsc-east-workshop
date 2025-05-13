# coordinator_agent.py

import os, uuid, json, requests, openai
from flask import Flask, request, jsonify
from typing import Optional
from common.types import (
    A2ARequest, SendTaskRequest, SendTaskResponse,
    Task, TaskStatus, TaskState,
    Message, TextPart, TaskSendParams
)

# ------------------------------------------------------------------
# OpenAI setup (requires openai>=1.0)
openai.api_key = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
# ------------------------------------------------------------------

SUMMARIZER_URL = "http://localhost:5001/tasks"
QNA_URL        = "http://localhost:5002/tasks"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "call_summarizer",
            "description": "Summarize long context into 2–3 sentences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_qna",
            "description": "Answer a question using the provided summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary":  {"type": "string"},
                    "question": {"type": "string"}
                },
                "required": ["summary", "question"]
            },
        },
    },
]

app = Flask("CoordinatorAgent")

# --------------------------- HTTP entry ---------------------------
@app.post("/tasks")
def receive_task():
    req_json = request.get_json(force=True)
    try:
        rpc_req: A2ARequest = A2ARequest.validate_python(req_json)
    except Exception as exc:
        return _rpc_error(req_json.get("id"), f"Invalid JSON-RPC: {exc}")

    if not isinstance(rpc_req, SendTaskRequest):
        return _rpc_error(rpc_req.id, "Only tasks/send supported")

    params = rpc_req.params
    parts  = params.message.parts or []
    if len(parts) < 2:
        return _task_fail(rpc_req.id, params.id,
                          "Need two parts: [context, question]")

    context, question = parts[0].text, parts[1].text

    # ----------------------------------------------------------------
    # NEW: keep track of intermediary results so we can return both
    # ----------------------------------------------------------------
    summary_cache: Optional[str] = None
    answer_cache:  Optional[str] = None

    # Chat with planner LLM
    messages = [
        {"role": "system",
         "content": ("You orchestrate tasks by calling the provided functions. "
                     "Use call_summarizer first for large contexts, then call_qna "
                     "with the summary and question. When done, reply with your "
                     "final answer as plain TEXT only.")},
        {"role": "user",
         "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]

    while True:
        try:
            resp = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=0
            )
        except Exception as exc:
            return _task_fail(rpc_req.id, params.id, f"OpenAI error: {exc}")

        choice = resp.choices[0]

        # ---- If LLM wants to call a tool ----
        if choice.finish_reason == "tool_calls":
            for call in choice.message.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)

                if fn_name == "call_summarizer":
                    summary = _delegate_worker(SUMMARIZER_URL, [args["text"]])
                    summary_cache = summary                              # <—
                    messages.extend([
                        choice.message,  # assistant w/ tool call
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": summary
                        }
                    ])

                elif fn_name == "call_qna":
                    answer = _delegate_worker(
                        QNA_URL, [args["summary"], args["question"]]
                    )
                    answer_cache = answer                                # <—
                    messages.extend([
                        choice.message,
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": answer
                        }
                    ])
            # Loop back to let LLM think again
            continue

        # ---- Final answer from LLM ----
        final_text = choice.message.content.strip()

        # ------------------------------------------------------------
        #  If we have both summary & answer, send them as TWO parts.
        #  Otherwise fall back to whatever the planner returned.
        # ------------------------------------------------------------
        if summary_cache and answer_cache:
            result_parts = [
                TextPart(text=summary_cache),
                TextPart(text=answer_cache)
            ]
        else:
            result_parts = [TextPart(text=final_text)]

        result_task = Task(
            id=params.id,
            status=TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(role="agent", parts=result_parts)
            )
        )
        return jsonify(SendTaskResponse(id=rpc_req.id,
                                        result=result_task).model_dump())

# ---------------------- helper functions --------------------------
def _delegate_worker(url: str, texts: list[str]) -> str:
    """Send a sub-task to summarizer/Q&A agent, return its text reply."""
    sub = TaskSendParams(
        id=uuid.uuid4().hex,
        message=Message(role="user",
                        parts=[TextPart(text=t) for t in texts])
    )
    payload = {"jsonrpc": "2.0", "id": 1,
               "method": "tasks/send", "params": sub.model_dump()}
    r = requests.post(url, json=payload, timeout=40)
    r.raise_for_status()
    task = Task.model_validate(r.json()["result"])
    return task.status.message.parts[0].text

def _rpc_error(rpc_id, msg):
    return jsonify({"jsonrpc":"2.0","id":rpc_id,
                    "error":{"code":-32600,"message":msg}})

def _task_fail(rpc_id, task_id, msg):
    failed = Task(
        id=task_id,
        status=TaskStatus(
            state=TaskState.FAILED,
            message=Message(role="agent",
                            parts=[TextPart(text=msg)])
        )
    )
    return jsonify(SendTaskResponse(id=rpc_id,
                                    result=failed).model_dump())

if __name__ == "__main__":
    app.run(port=5000)
