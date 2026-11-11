"""Load the identity-specific annotation prompt (Figure S5) in place of the hypothesaes default."""

from pathlib import Path

ANNOTATE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "annotate.txt"


def use_identity_specific_prompt():
    """Make hypothesaes use prompts/annotate.txt for fidelity scoring and annotation."""
    import hypothesaes.annotate as _ann
    from hypothesaes.utils import load_prompt as stock_load_prompt

    prompt = ANNOTATE_PROMPT_PATH.read_text()
    _ann.load_prompt = lambda name: prompt if name == "annotate" else stock_load_prompt(name)
