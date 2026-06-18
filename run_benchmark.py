from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_cost_table, save_report
from src.reflexion_lab.runtime import IS_MOCK
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(dataset: str = "data/hotpot_dev_qa.json", out_dir: str = "outputs/sample_run", reflexion_attempts: int = 3, limit: int = 0) -> None:
    examples = load_dataset(dataset)
    if limit > 0:
        examples = examples[:limit]
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode="mock" if IS_MOCK else "gemini")
    json_path, md_path = save_report(report, out_path)
    cost_path = save_cost_table(all_records, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(f"[green]Saved[/green] {cost_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
