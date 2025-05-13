# qna_agent.py
import os, openai
from flask import Flask, request, jsonify
from common.types import (
    A2ARequest, SendTaskRequest, SendTaskResponse,
    Task, TaskStatus, TaskState,
    Message, TextPart
)

openai.api_key = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = Flask("QnAAgent")

@app.post("/tasks")
def handle_task():
    req_json = request.get_json(force=True)
    try:
        rpc_req: A2ARequest = A2ARequest.validate_python(req_json)
    except Exception as exc:
        return _rpc_err(req_json.get("id"), f"Invalid JSON-RPC: {exc}")

    if not isinstance(rpc_req, SendTaskRequest):
        return _rpc_err(rpc_req.id, "Unsupported method")

    params = rpc_req.params
    parts  = params.message.parts or []
    if len(parts) < 2:
        return _task_fail(rpc_req.id, params.id, "Need context + question")

    context, question = parts[0].text, parts[1].text
    answer = _answer(context, question) or "(no answer returned)"

    done_task = Task(
        id=params.id,
        status=TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text=answer)])
        )
    )
    return jsonify(SendTaskResponse(id=rpc_req.id, result=done_task).model_dump())

# ----------------------- helpers ----------------------------------

def _answer(context: str, question: str) -> str:
    """Call OpenAI chat completion and always return a non-None string."""
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant. Answer strictly using the provided context."},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user",   "content": question}
    ]
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=256
        )
        # Safely extract the assistant’s reply
        choice = resp.choices[0] if resp.choices else None
        text = choice.message.content if choice and choice.message.content else ""
        
        answer = text.strip()

        # Print question and answer for debugging purposes
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("-" * 40)

        return answer
    except Exception:
        # On API or parsing errors, return an empty string to avoid None
        return ""

def _task_fail(rpc_id, task_id, msg):
    failed = Task(
        id=task_id,
        status=TaskStatus(
            state=TaskState.FAILED,
            message=Message(role="agent", parts=[TextPart(text=msg)])
        )
    )
    return jsonify(SendTaskResponse(id=rpc_id, result=failed).model_dump())

def _rpc_err(rpc_id, msg):
    return jsonify({"jsonrpc": "2.0", "id": rpc_id,
                    "error": {"code": -32600, "message": msg}})

if __name__ == "__main__":
    app.run(port=5002)
