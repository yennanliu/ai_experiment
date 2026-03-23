"""Job management and output handling."""

import json
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from agent_team.logger import get_logger


logger = get_logger(__name__)


class JobManager:
    """Manages job execution, output paths, and artifacts."""

    def __init__(
        self,
        base_output_dir: Path = Path("output"),
        base_log_dir: Path = Path("log"),
    ):
        """Initialize job manager.

        Args:
            base_output_dir: Base directory for outputs (default: ./output)
            base_log_dir: Base directory for logs (default: ./log)
        """
        self.base_output_dir = base_output_dir
        self.base_log_dir = base_log_dir
        self.job_id = self._generate_job_id()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        return str(uuid.uuid4())[:8]

    @property
    def output_dir(self) -> Path:
        """Get the output directory for this job."""
        path = self.base_output_dir / self.timestamp / self.job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        """Get the log directory for this job."""
        path = self.base_log_dir / self.timestamp / self.job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_output(self, filename: str, content: str) -> Path:
        """Save content to output file.

        Args:
            filename: Name of the output file
            content: Content to save

        Returns:
            Path to the saved file
        """
        filepath = self.output_dir / filename
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Saved output: {filepath}")
        return filepath

    def save_json(self, filename: str, data: dict) -> Path:
        """Save JSON data to file.

        Args:
            filename: Name of the JSON file
            data: Dictionary to save as JSON

        Returns:
            Path to the saved file
        """
        filepath = self.output_dir / filename
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved JSON: {filepath}")
        return filepath

    def save_execution_metadata(self, task: str, state) -> Path:
        """Save execution metadata and results.

        Args:
            task: The task that was executed
            state: WorkflowState object with results

        Returns:
            Path to the saved metadata file
        """
        metadata = {
            "job_id": self.job_id,
            "timestamp": self.timestamp,
            "task": task,
            "pattern": state.pattern.value,
            "agents": [r.value for r in state.agents_executed],
            "total_tokens": state.total_tokens,
            "success": state.success,
            "error": state.error,
            "responses": [
                {
                    "role": r.role.value,
                    "tokens_used": r.tokens_used,
                    "success": r.success,
                    "error": r.error,
                }
                for r in state.responses
            ],
        }
        return self.save_json("metadata.json", metadata)

    def save_full_output(self, state) -> Path:
        """Save full detailed output with all responses.

        Args:
            state: WorkflowState object with results

        Returns:
            Path to the saved file
        """
        output = []
        output.append(f"{'='*80}")
        output.append("EXECUTION SUMMARY")
        output.append(f"{'='*80}")
        output.append(f"Job ID: {self.job_id}")
        output.append(f"Timestamp: {self.timestamp}")
        output.append(f"Pattern: {state.pattern.value}")
        output.append(f"Agents: {', '.join(r.value for r in state.agents_executed)}")
        output.append(f"Total Tokens: {state.total_tokens:,}")
        output.append(f"Success: {state.success}")
        if state.error:
            output.append(f"Error: {state.error}")
        output.append(f"{'='*80}\n")

        for i, response in enumerate(state.responses):
            output.append(f"{'─'*80}")
            output.append(f"AGENT {i+1}: {response.role.value.upper()}")
            output.append(f"{'─'*80}")
            output.append(f"Tokens: {response.tokens_used}")
            output.append(f"Success: {response.success}")
            if response.error:
                output.append(f"Error: {response.error}")
            output.append("")
            output.append(response.content)
            output.append("")

        content = "\n".join(output)
        return self.save_output("output.txt", content)

    def get_job_info(self) -> dict:
        """Get information about the current job.

        Returns:
            Dictionary with job information
        """
        return {
            "job_id": self.job_id,
            "timestamp": self.timestamp,
            "output_dir": str(self.output_dir),
            "log_dir": str(self.log_dir),
        }
