# NUS ISS AI Agent Workshop 03 - Group 05

## 1. Project Overview
This is a multi-agent primary school tutoring system built with **Python + LangGraph**.

The system has 4 roles:
- `orchestrator`: uses an LLM-based routing decision to choose which agent to call next, and decides when to stop after multi-step coordination.
- `Agent-01 (Chinese Tutor)`: private Chinese tutor, only allowed to use `get_poem_tool`.
- `Agent-02 (Math Tutor)`: private Math tutor, not allowed to use any external tool.
- `Agent-03 (Weekly Planner)`: weekly study planner, only allowed to use `get_date_tool`.

Workflow:
`user input -> orchestrator -> agent(s) -> orchestrator -> ... -> finish`

The project uses `session_id` for session-level state management and context passing across nodes.

## 2. Tools and Permission Isolation
- `get_poem_tool`: calls `https://v1.jinrishici.com/all.json` to fetch a random Chinese poem line.
- `get_date_tool`: uses Python `datetime` to return current date and time.

Hard permission isolation:
- Chinese Agent binds only `get_poem_tool`
- Planner Agent binds only `get_date_tool`
- Math Agent binds no tools
- Orchestrator does not call tools

Routing behavior:
- Orchestrator is LLM-driven (not keyword-only matching).
- Orchestrator chooses one route per decision: `chinese`, `math`, `planner`, or `finish`.

## 3. Project Structure
```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── workshop03_multi_agent.ipynb
├── docs/
│   └── permission-proof.md
└── src/
    └── tutor_multi_agent/
        ├── __init__.py
        ├── cli.py
        ├── graph.py
        ├── runner.py
        ├── state.py
        └── tools.py
```

## 4. Full Run Commands
```bash
# 1) Create and activate conda environment (adjust Python version if needed)
conda create -n workshop03 python=3.11 -y
conda activate workshop03

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set environment variables
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 4) Start Notebook
jupyter notebook notebooks/workshop03_multi_agent.ipynb
```

## 5. Quick CLI Verification
```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

python -m tutor_multi_agent.cli \
  --session demo-chinese \
  --input "Please provide one Chinese poem line and explain its meaning."

python -m tutor_multi_agent.cli \
  --session demo-math \
  --input "For a Grade 3 student, how do we solve 23 + 19 step by step?"

python -m tutor_multi_agent.cli \
  --session demo-plan \
  --input "Please create a weekly study plan based on the current date."
```

## 6. Core Input Examples (3-4)
1. `Give me one classical Chinese poem line and explain it like a primary school teacher.`
2. `For a Grade 3 student, how do we solve 45 - 18 step by step?`
3. `Based on today's date, create a weekly study plan for Chinese and Math.`
4. `First share one poem line, then give me a Chinese and Math review plan for this week.`

Example 4 demonstrates cross-agent routing by the orchestrator.

## 7. Permission Proof (Runtime Log Screenshot)
1. Open the notebook and run the "Permission proof" cell.
2. Check `tool_events` in the output:
   - Chinese scenario may include `get_poem_tool`
   - Chinese scenario must not include `get_date_tool`
3. Save the output screenshot as: `docs/permission-proof.png`

Reference guide: [docs/permission-proof.md](docs/permission-proof.md)

## 8. Notes
- All prompts and code comments are in English.
- `OPENAI_API_KEY` is read only from environment variables.
- No extra telemetry or unrelated network calls are added.
