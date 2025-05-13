# client.py
"""
Simple CLI client for the Coordinator agent.
Usage:
    python client.py "long context here" "question here"
If no args are supplied, a default Artemis example is used.
"""
import sys, uuid, requests, json
from common.types import Message, TextPart, TaskSendParams

COORD_URL = "http://localhost:5000/tasks"
JSONRPC_ID = 1

def build_payload(context: str, question: str):
    msg = Message(
        role="user",
        parts=[TextPart(text=context), TextPart(text=question)]
    )
    params = TaskSendParams(id=uuid.uuid4().hex, message=msg)
    return {
        "jsonrpc": "2.0",
        "id": JSONRPC_ID,
        "method": "tasks/send",
        "params": params.model_dump()
    }

def send(context: str, question: str):
    payload = build_payload(context, question)
    resp = requests.post(COORD_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        print("Coordinator error:", json.dumps(data["error"], indent=2))
        return

    parts = (data["result"]
             ["status"]["message"]["parts"])

    # Flexible printing: one-part or two-part response
    if len(parts) == 1:
        print("\nANSWER:\n", parts[0]["text"])
    else:
        print("\nSUMMARY:\n", parts[0]["text"])
        print("\nANSWER:\n", parts[1]["text"])

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        ctx, q = sys.argv[1], sys.argv[2]
    else:
        ctx = (
            """From the moment I first read the mission statement—“NASA's Artemis program aims to establish a sustainable human presence on the Moon”—I felt like a kid again, staring up at the night sky and daring to dream. Back then, my only lunar ambition was convincing my parents I really needed that telescope for my birthday. Today, I'm one of the engineers wrestling with that very statement, unpacking every word to turn it into reality. My coffee-fueled mornings begin with whiteboards scrawled in orbital mechanics and habitat schematics, each doodle a stubborn reminder that sustaining life on the Moon is more than planting a flag and taking selfies.

            Transporting large payloads isn't as cinematic as in the movies—there are no dramatic launch sequences with glowing thrusters that hover above a lunar cliff. Instead, there's a delicate ballet of mass budgets, structural stiffness, and propulsion trade-offs. I remember when our team tested a prototype cargo lander's legs in the desert; watching that machine wobble under load was like watching a toddler try to stand after one too many birthday cakes. We redesigned the leg geometry three times because telescoping struts that seemed rock-solid on paper kept bending like wet spaghetti under real-world stresses.

            Then comes the marathon of long-term life-support. It's one thing to keep a plant alive in a lab for a week, quite another to ensure a greenhouse under lunar regolith thrives for months. I've spent countless hours analyzing water-recycling loops and atmospheric scrubbers, peering at sensor readouts that hint at biological processes more temperamental than my houseplants back on Earth. Every hiccup—a droplet of condensate where it shouldn't be, a CO₂ spike at 3 a.m.—sparks spirited debates over algorithm tweaks and hardware tweaks. Truth be told, there's a certain thrill in chasing down these microscopic gremlins; it's like gardening in zero gravity.

            Shielding astronauts from cosmic radiation might sound like wrapping them in lead blankets, but the real solution is far more elegant—and far more complex. We're experimenting with regolith-packed walls, magnetic deflection fields, and even polyethylene composites that absorb high-energy particles. I still chuckle at my first simulation, where a single solar flare lobbed enough radiation at our habitat model to trigger alarms louder than a 1980s arcade game. Those early failures taught us humility: Mother Nature will always find a way to remind you that you're not the smartest one in the room.

            Looking back, designing for Artemis has been the most exhilarating engineering challenge of my career. Every setback has sharpened our ingenuity, every late-night breakthrough reaffirmed why we chose this path. And here's my call to action: whether you're a student sketching rockets on a napkin or a seasoned researcher tweaking life-support membranes, lean in. The Moon is waiting not for spectators, but for problem-solvers ready to turn “what if” into “what next.” Let's make Artemis the beginning of humanity's greatest adventure yet."""
        )
        q = (
            "What is one major engineering challenge not mentioned in the text provided?"
        )
    send(ctx, q)
