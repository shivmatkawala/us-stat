"""
Local data-agent loop around an Azure AI Foundry agent.

Flow: scan CSVs -> agent writes code -> you approve -> code runs locally
      -> output/errors go back to the agent -> it debugs -> repeat.

Setup:  pip install azure-ai-projects>=2.1.0 azure-identity pandas matplotlib seaborn
Run:    python frosty_data_agent.py            (from the folder containing your CSVs)
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# ---------- config ----------
ENDPOINT = "https://foundry-da-173.services.ai.azure.com/api/projects/project-default"
AGENT_NAME = "frosty-agent-qk4sm26tl0"
AGENT_VERSION = "1"

DATA_DIR = Path.cwd()
WORK_DIR = DATA_DIR / "agent_work"      # generated scripts live here
OUT_DIR = DATA_DIR / "output"           # cleaned data, dim/fact tables, charts
MAX_AUTO_FIXES = 3                      # consecutive failed runs before stopping
RUN_TIMEOUT_SEC = 180
MAX_OUTPUT_CHARS = 4000
SAMPLE_ROWS = 3                         # rows per file sent to the agent

RISKY = re.compile(
    r"\b(os\.remove|os\.unlink|shutil\.rmtree|rmtree|subprocess|os\.system|"
    r"requests\.|urllib|socket|pip install|\.unlink\(|os\.rmdir)\b"
)

PROTOCOL = f"""
You are a data engineering assistant working on the user's LOCAL machine.
I run your code for you and send back stdout/stderr.

RULES
1. To run code, reply with EXACTLY ONE ```python block, with at most 2 sentences of explanation before it.
2. Use only pandas, numpy, matplotlib, seaborn and the standard library.
3. Read inputs from the current working directory. Write ALL outputs (cleaned CSVs, dim_*.csv, fact_*.csv,
   PNG charts) into the folder "output" (create it if missing). Never modify or delete the original files.
4. Never call plt.show(). Use matplotlib.use("Agg") and plt.savefig("output/<name>.png", dpi=150, bbox_inches="tight").
5. Print useful evidence: shapes, null counts, dtypes, head(), row counts before/after, and key checks
   (e.g. unique keys, fact-to-dim join integrity).
6. Work in small steps: profile -> clean -> build dims -> build fact -> dashboard. One step per code block.
7. If I send you an error, find the root cause, explain it in 1-2 sentences, then send corrected full code.
8. If you need a decision from me (e.g. how to treat nulls, which column is the grain), ask in plain text
   WITHOUT a code block and wait for my answer.
9. When a task is finished, reply in plain text (no code block) with a short summary and list the files created.
Output folder: {OUT_DIR.name}
""".strip()


# ---------- scanning ----------
def scan_files() -> str:
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        return "No CSV files found in the current folder."
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=2000, encoding_errors="replace")
            parts.append(
                f"FILE: {f.name}\n"
                f"columns/dtypes (from first 2000 rows):\n{df.dtypes.to_string()}\n"
                f"sample rows:\n{df.head(SAMPLE_ROWS).to_string(max_colwidth=40)}\n"
            )
        except Exception as e:  # noqa: BLE001
            parts.append(f"FILE: {f.name}  (could not read: {e})")
    return "\n".join(parts)


# ---------- agent ----------
class Agent:
    def __init__(self):
        project = AIProjectClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
        self.client = project.get_openai_client()
        self.history: list[dict] = []

    def send(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        resp = self.client.responses.create(
            input=self.history,
            extra_body={
                "agent_reference": {
                    "name": AGENT_NAME,
                    "version": AGENT_VERSION,
                    "type": "agent_reference",
                }
            },
        )
        reply = resp.output_text or ""
        self.history.append({"role": "assistant", "content": reply})
        return reply


# ---------- helpers ----------
def extract_code(reply: str) -> str | None:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply, re.DOTALL)
    return m.group(1).strip() if m else None


def ask_consent(code: str, step: int) -> bool:
    print("\n" + "=" * 70)
    print(f"PROPOSED CODE (step {step})")
    print("=" * 70)
    print(code)
    print("=" * 70)
    if RISKY.search(code):
        print("WARNING: this code contains potentially risky operations "
              "(file deletion, network, subprocess). Review carefully.")
    while True:
        ans = input("Run this code? [y]es / [n]o : ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def run_code(code: str, step: int) -> tuple[bool, str]:
    WORK_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    script = WORK_DIR / f"step_{step:02d}.py"
    script.write_text(code, encoding="utf-8")
    try:
        p = subprocess.run(
            [sys.executable, str(script)],
            cwd=DATA_DIR, capture_output=True, text=True,
            timeout=RUN_TIMEOUT_SEC, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: script ran longer than {RUN_TIMEOUT_SEC}s and was stopped."
    out = p.stdout[-MAX_OUTPUT_CHARS:]
    err = p.stderr[-MAX_OUTPUT_CHARS:]
    ok = p.returncode == 0
    report = f"exit_code={p.returncode}\n--- STDOUT ---\n{out}\n--- STDERR ---\n{err}"
    return ok, report


# ---------- main loop ----------
def main():
    agent = Agent()
    print(f"Scanning CSV files in {DATA_DIR} ...")
    summary = scan_files()
    print(summary)
    print("\nNOTE: column names and a few sample rows (plus code output) are sent to your Azure agent.\n")

    goal = input("What do you want to do? (e.g. 'clean these files, build star schema, make a dashboard')\n> ").strip()
    if not goal:
        goal = "Profile the data, clean it, build dim and fact tables, then create a dashboard."

    reply = agent.send(f"{PROTOCOL}\n\nFILES IN WORKING FOLDER:\n{summary}\n\nMY GOAL:\n{goal}")
    step, fails = 0, 0

    while True:
        code = extract_code(reply)
        text_only = re.sub(r"```.*?```", "", reply, flags=re.DOTALL).strip()
        if text_only:
            print(f"\nAGENT: {text_only}")

        if code is None:
            # agent is asking a question or finished
            nxt = input("\nYou (Enter to quit): ").strip()
            if not nxt:
                print("Done. Outputs are in:", OUT_DIR)
                return
            reply, fails = agent.send(nxt), 0
            continue

        step += 1
        if not ask_consent(code, step):
            fb = input("Tell the agent what to change (Enter to stop): ").strip()
            if not fb:
                print("Stopped.")
                return
            reply = agent.send(f"I declined to run that code. {fb}")
            continue

        ok, report = run_code(code, step)
        print("\n--- EXECUTION RESULT ---")
        print(report)

        if ok:
            fails = 0
            reply = agent.send(f"Execution succeeded.\n{report}\nContinue with the next step, or summarize if finished.")
        else:
            fails += 1
            print(f"\nRUN FAILED ({fails}/{MAX_AUTO_FIXES})")
            if fails >= MAX_AUTO_FIXES:
                print("The agent could not fix this automatically. Asking it for a plain-language diagnosis...")
                reply = agent.send(
                    f"Execution failed again.\n{report}\n"
                    "Stop writing code. In plain text (no code block) explain what is going wrong, "
                    "the most likely root cause, and what I should check or decide."
                )
                fails = 0
                continue
            reply = agent.send(f"Execution FAILED.\n{report}\nDiagnose the root cause and send corrected full code.")


if __name__ == "__main__":
    main()