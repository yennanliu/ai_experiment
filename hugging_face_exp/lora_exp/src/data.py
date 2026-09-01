"""A tiny, hand-written training set.

Small enough that a 82M-parameter model can visibly overfit it in ~60 steps on
a laptop CPU, which is exactly what we want: proof that the LoRA adapter (and
only the LoRA adapter) learned something.
"""

# The style we are teaching: every answer starts with "Fact:" and is one line.
EXAMPLES = [
    "Q: What is LoRA?\nA: Fact: LoRA freezes the base weights and trains two small matrices A and B.",
    "Q: What does the rank r control?\nA: Fact: r sets the width of the bottleneck, so it sets how many parameters you train.",
    "Q: What is alpha for?\nA: Fact: alpha scales the adapter output by alpha divided by r.",
    "Q: Which layers get an adapter?\nA: Fact: usually the attention projections, q_proj and v_proj first.",
    "Q: What is QLoRA?\nA: Fact: QLoRA keeps the frozen base in 4-bit and trains the adapters in higher precision.",
    "Q: How big is an adapter?\nA: Fact: an adapter is typically under one percent of the base model.",
    "Q: Can adapters be merged?\nA: Fact: merging adds B times A times the scaling into the base weight, so inference costs nothing extra.",
    "Q: Why not full fine-tuning?\nA: Fact: full fine-tuning needs optimizer state for every weight, which is the memory that blows up.",
]

# Prompts held out of training, used to eyeball whether the style generalizes.
EVAL_PROMPTS = [
    "Q: What is LoRA?\nA:",
    "Q: What does a low-rank matrix do?\nA:",
]
