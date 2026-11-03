"""SGLang model package of the HEED student (qwen3_vl.py needs SGLang; this file does not)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def serve_students(model_path):
    """Let SGLang, in this process and the ones it spawns, load `model_path` if it is an exported HEED student.

    Call before SGLang is imported; a no-op for other checkpoints, which SGLang then serves with its own classes.
    """
    if os.path.isfile(os.path.join(model_path, "configuration_heed.py")):
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = __name__
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")]))
