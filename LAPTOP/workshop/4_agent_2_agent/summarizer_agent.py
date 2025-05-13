# summarizer_agent.py
import os, openai
from flask import Flask, request, jsonify
from common.types import (
    A2ARequest, SendTaskRequest, SendTaskResponse,
    Task, TaskStatus, TaskState,
    Message, TextPart
)

openai.api_key  = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = Flask("SummarizerAgent")

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
    text_to_sum = params.message.parts[0].text if params.message.parts else ""
    summary = _summarize(text_to_sum)

    done_task = Task(
        id=params.id,
        status=TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text=summary)])
        )
    )
    return jsonify(SendTaskResponse(id=rpc_req.id, result=done_task).model_dump())

# ----------------------- helpers ----------------------------------

def _summarize(text: str) -> str:
    prompt = f"Summarize the following text in 2-3 sentences:\n\n{text}"
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    
    answer = resp.choices[0].message.content.strip()

    # print question and answer for debugging purposes
    print(f"Query: {text}")
    print(f"Summary: {answer}")
    print("-" * 40)

    return answer

def _rpc_err(rpc_id, msg):
    return jsonify({"jsonrpc": "2.0", "id": rpc_id,
                    "error": {"code": -32600, "message": msg}})

if __name__ == "__main__":
    app.run(port=5001)
