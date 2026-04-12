"""Colored terminal logger for the ticket pipeline"""

import logging
import sys

# ANSI colors
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
NODE_COLOR  = "\033[94m"   # bright blue  — node name
TICK_COLOR  = "\033[93m"   # bright yellow — ticket id
LLM_COLOR   = "\033[90m"   # grey — LLM payload


class _PipelineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level_color = COLORS.get(record.levelname, "")
        level_tag   = f"{level_color}{BOLD}[{record.levelname[0]}]{RESET}"
        ts          = self.formatTime(record, "%H:%M:%S")
        ts_str      = f"{DIM}{ts}{RESET}"

        # Extra fields injected by pipeline nodes
        node    = getattr(record, "node", None)
        ticket  = getattr(record, "ticket_id", None)
        llm_in  = getattr(record, "llm_in", None)
        llm_out = getattr(record, "llm_out", None)

        parts = [level_tag, ts_str]
        if ticket:
            parts.append(f"{TICK_COLOR}{ticket}{RESET}")
        if node:
            parts.append(f"{NODE_COLOR}[{node}]{RESET}")
        parts.append(record.getMessage())

        line = "  ".join(parts)

        if llm_in:
            preview = llm_in.replace("\n", " ")[:120]
            line += f"\n    {LLM_COLOR}→ prompt : {preview}{RESET}"
        if llm_out:
            preview = llm_out.replace("\n", " ")[:120]
            line += f"\n    {LLM_COLOR}← response: {preview}{RESET}"

        return line


def get_logger(name: str = "ticket") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_PipelineFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


log = get_logger()
