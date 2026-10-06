from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

ENDPOINT = "https://foundry-da-173.services.ai.azure.com/api/projects/project-default"
AGENT_NAME = "frosty-agent-qk4sm26tl0"
AGENT_VERSION = "1"

project_client = AIProjectClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
openai_client = project_client.get_openai_client()

history = []  # full conversation so the agent remembers context

print("Chat with Frosty. Type 'exit' to quit, 'reset' to start a new conversation.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        break

    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit", "q"):
        print("Bye!")
        break
    if user_input.lower() == "reset":
        history.clear()
        print("Conversation cleared.\n")
        continue

    history.append({"role": "user", "content": user_input})
    try:
        response = openai_client.responses.create(
            input=history,
            extra_body={
                "agent_reference": {
                    "name": AGENT_NAME,
                    "version": AGENT_VERSION,
                    "type": "agent_reference",
                }
            },
        )
    except Exception as e:
        history.pop()  # drop the failed message so history stays consistent
        print(f"\n[Error] {e}\n")
        continue

    reply = response.output_text
    history.append({"role": "assistant", "content": reply})
    print(f"\nFrosty: {reply}\n")