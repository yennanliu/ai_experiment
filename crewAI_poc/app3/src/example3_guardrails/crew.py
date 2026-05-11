import os
from datetime import datetime
from crewai import Crew, LLM, Process, TaskOutput
from src.example3_guardrails.agents import analyst, writer
from src.example3_guardrails.tasks import analysis_task, report_task

LOG_DIR = "output"


def _log(filename: str, message: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, filename), "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}\n")


def build_crew(topic: str) -> Crew:
    llm = LLM(model="openai/gpt-4o-mini")
    _analyst = analyst(llm)
    _writer = writer(llm)
    t1 = analysis_task(_analyst, topic)
    t2 = report_task(_writer, t1, topic)

    def on_step(step) -> None:
        _log("steps.log", f"{type(step).__name__} | {str(step)[:200]}")

    def on_task(output: TaskOutput) -> None:
        words = len(output.raw.split())
        _log(
            "tasks.log",
            f"DONE | agent={output.agent} | words={words} | guardrail=PASSED",
        )
        print(f"  [task_callback] agent={output.agent} | {words} words | guardrail PASSED")

    return Crew(
        agents=[_analyst, _writer],
        tasks=[t1, t2],
        process=Process.sequential,
        step_callback=on_step,
        task_callback=on_task,
        verbose=True,
    )
