"""
Frosty file agent: the Foundry agent can READ, WRITE and RUN files in your project,
but every action needs your approval. Writes show a diff first and keep a backup.

Put this file in your frosty/ folder and run:  python frosty/frosty_file_agent.py
"""
import difflib
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# ---------- config ----------
ENDPOINT = "https://foundry-da-173.services.ai.azure.com/api/projects/project-default"
AGENT_NAME = "frosty-agent-qk4sm26tl0"
AGENT_VERSION = "1"

PROJECT_DIR = Path(__file__).resolve().parent.parent   # us-da-stat (parent of frosty/)
BACKUP_DIR = PROJECT_DIR / "agent_backups"
SKIP_DIRS = {"frosty-venv", ".git", "__pycache__", "agent_backups", "node_modules", ".venv"}
WRITE_EXT = {".py", ".md", ".txt", ".json", ".sql", ".yaml", ".yml"}
READ_EXT = WRITE_EXT | {".csv", ".tsv"}
MAX_READ_CHARS = 6000        # how much of a file is sent to the agent
MAX_OUTPUT_CHARS = 4000
RUN_TIMEOUT_SEC = 180
MAX_AUTO_FIXES = 3

PROTOCOL = """
You are a data engineering assistant working on the user's LOCAL project. I execute your requests after the user approves.

ACTIONS (use only these formats):
1. WRITE a file: a fenced block whose opening line is  ```python file=<relative/path.py>
   Always send the COMPLETE file content, never fragments. Paths are relative to the project root.
2. READ a file: a line by itself:  READ: <relative/path>
   Use this to see current code or the head of a data file before changing things.
3. RUN a python file: a line by itself:  RUN: <relative/path.py>
   I will send you stdout/stderr.

RULES
- Do not guess file contents; READ first if you need them.
- Use pandas, numpy, matplotlib, seaborn and the standard library only.
- Never modify or delete files under 'downloads/' (raw data). Write outputs (clean data, dim/fact tables, PNG charts) to 'output/'.
- Charts: matplotlib.use("Agg"), savefig to output/, never plt.show().
- Make small steps. You may combine WRITE + RUN in one reply.
- If I send you an error, explain the root cause in 1-2 sentences, then send the corrected complete file.
- If you need a decision from me, ask in plain text with no actions.
- When finished, reply in plain text summarizing what was created.
""".strip()

FILE_BLOCK = re.compile(r"```[a-zA-Z]*[ \t]+file=([^\s`]+)[ \t]*\n(.*?)```", re.DOTALL)
READ_LINE = re.compile(r"^READ:\s*(.+?)\s*$", re.MULTILINE)
RUN_LINE = re.compile(r"^RUN:\s*(.+?)\s*$", re.MULTILINE)


# ---------- helpers ----------
def safe_path(rel: str, allowed_ext: set[str]) -> Path | None:
    """Resolve a path and make sure it stays inside the project and is not in a skipped folder."""
    p = (PROJECT_DIR / rel.strip().strip("'\"")).resolve()
    try:
        parts = p.relative_to(PROJECT_DIR).parts
    except ValueError:
        return None
    if any(part in SKIP_DIRS for part in parts) or p.suffix.lower() not in allowed_ext:
        return None
    return p


def project_tree() -> str:
    lines = []
    for p in sorted(PROJECT_DIR.rglob("*")):
        rel = p.relative_to(PROJECT_DIR)
        if any(part in SKIP_DIRS for part in rel.parts) or p.is_dir():
            continue
        lines.append(f"{rel.as_posix()}  ({p.stat().st_size} bytes)")
    return "\n".join(lines[:200]) or "(empty project)"


def yes(prompt: str) -> bool:
    while True:
        a = input(f"{prompt} [y/n]: ").strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False


class Agent:
    def __init__(self):
        project = AIProjectClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
        self.client = project.get_openai_client()
        self.history: list[dict] = []

    def send(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        resp = self.client.responses.create(
            input=self.history,
            extra_body={"agent_reference": {"name": AGENT_NAME, "version": AGENT_VERSION, "type": "agent_reference"}},
        )
        reply = resp.output_text or ""
        self.history.append({"role": "assistant", "content": reply})
        return reply


# ---------- actions ----------
def do_write(rel: str, content: str) -> str:
    p = safe_path(rel, WRITE_EXT)
    if p is None or "downloads" in p.relative_to(PROJECT_DIR).parts:
        return f"WRITE {rel}: REFUSED (path outside project, protected folder, or file type not allowed)."
    old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    new = content.rstrip() + "\n"
    if old == new:
        return f"WRITE {rel}: no changes (identical content)."

    print("\n" + "=" * 70)
    print(f"WRITE REQUEST: {p.relative_to(PROJECT_DIR).as_posix()}  ({'new file' if not p.exists() else 'modify existing'})")
    print("=" * 70)
    if old:
        diff = difflib.unified_diff(old.splitlines(), new.splitlines(), "current", "proposed", lineterm="")
        print("\n".join(diff))
    else:
        print(new)
    if not yes("Write this file?"):
        return f"WRITE {rel}: DECLINED by user."

    if p.exists():
        BACKUP_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (BACKUP_DIR / f"{p.stem}_{stamp}{p.suffix}.bak").write_text(old, encoding="utf-8")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new, encoding="utf-8")
    print(f"Saved {p.relative_to(PROJECT_DIR).as_posix()}")
    return f"WRITE {rel}: saved."


def do_read(rel: str) -> str:
    p = safe_path(rel, READ_EXT)
    if p is None or not p.exists():
        return f"READ {rel}: REFUSED or not found."
    if not yes(f"Agent wants to READ '{rel}' (first {MAX_READ_CHARS} chars are sent to Azure). Allow?"):
        return f"READ {rel}: DECLINED by user."
    text = p.read_text(encoding="utf-8", errors="replace")[:MAX_READ_CHARS]
    return f"READ {rel}:\n```\n{text}\n```"


def do_run(rel: str) -> tuple[bool, str]:
    p = safe_path(rel, {".py"})
    if p is None or not p.exists():
        return False, f"RUN {rel}: REFUSED or not found."
    if not yes(f"Run 'python {p.relative_to(PROJECT_DIR).as_posix()}' now?"):
        return True, f"RUN {rel}: DECLINED by user."
    try:
        r = subprocess.run(
            [sys.executable, str(p)], cwd=PROJECT_DIR, capture_output=True, text=True,
            timeout=RUN_TIMEOUT_SEC, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, f"RUN {rel}: TIMEOUT after {RUN_TIMEOUT_SEC}s."
    report = (f"RUN {rel}: exit_code={r.returncode}\n--- STDOUT ---\n{r.stdout[-MAX_OUTPUT_CHARS:]}"
              f"\n--- STDERR ---\n{r.stderr[-MAX_OUTPUT_CHARS:]}")
    print("\n--- EXECUTION RESULT ---\n" + report)
    return r.returncode == 0, report


# ---------- main ----------
def main():
    agent = Agent()
    print(f"Project root: {PROJECT_DIR}")
    tree = project_tree()
    print("Files visible to the agent:\n" + tree)
    print("\nNote: file names, requested file contents and run output are sent to your Azure agent.\n")

    goal = input("What should Frosty do?\n> ").strip()
    reply = agent.send(f"{PROTOCOL}\n\nPROJECT FILES:\n{tree}\n\nMY REQUEST:\n{goal}")
    fails = 0

    while True:
        writes = FILE_BLOCK.findall(reply)
        reads = READ_LINE.findall(reply)
        runs = RUN_LINE.findall(reply)
        prose = FILE_BLOCK.sub("", reply)
        prose = RUN_LINE.sub("", READ_LINE.sub("", prose)).strip()
        if prose:
            print(f"\nFROSTY: {prose}")

        if not (writes or reads or runs):
            nxt = input("\nYou (Enter to quit): ").strip()
            if not nxt:
                print("Done. Backups of overwritten files are in:", BACKUP_DIR)
                return
            reply, fails = agent.send(nxt), 0
            continue

        results, run_failed = [], False
        for rel, content in writes:
            results.append(do_write(rel, content))
        for rel in reads:
            results.append(do_read(rel))
        for rel in runs:
            ok, report = do_run(rel)
            results.append(report)
            run_failed = run_failed or not ok

        if run_failed:
            fails += 1
            if fails >= MAX_AUTO_FIXES:
                print(f"\nRun failed {fails} times in a row. Asking the agent for a plain-language diagnosis...")
                reply = agent.send("\n\n".join(results) + "\n\nStop writing code. Explain in plain text what is going "
                                   "wrong, the likely root cause, and what I should check or decide.")
                fails = 0
                continue
            print(f"\nRUN FAILED ({fails}/{MAX_AUTO_FIXES}) - sending the error to the agent to fix.")
            reply = agent.send("\n\n".join(results) + "\n\nThe run FAILED. Diagnose and send the corrected full file.")
        else:
            fails = 0
            reply = agent.send("\n\n".join(results) + "\n\nContinue, or summarize if finished.")


if __name__ == "__main__":
    main()