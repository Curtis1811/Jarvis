"""Hide harmless start-up warnings from the libraries behind the local voices.

They are notices meant for developers ("redirects are not supported on
Windows", "you are sending unauthenticated requests to the HF Hub", ...), not
problems, and they bury what Jarvis prints. Call quiet() before loading models.
"""

from __future__ import annotations

import logging
import os
import warnings

_HIDDEN = (
    "Redirects are currently not supported",
    "unauthenticated requests to the HF Hub",
    "Defaulting repo_id to",
    "dropout option adds dropout after all but last recurrent layer",
    "`torch.nn.utils.weight_norm` is deprecated",
    "symlinks",
)


class _Hide(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(text in message for text in _HIDDEN)


_done = False


def quiet() -> None:
    global _done
    if _done:
        return
    _done = True
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for text in _HIDDEN:
        warnings.filterwarnings("ignore", message=f".*{text}.*")
    # A filter on the logger survives the libraries resetting its level.
    hide = _Hide()
    for name in ("torch.distributed.elastic.multiprocessing.redirects", "huggingface_hub",
                 "huggingface_hub.utils._http", "huggingface_hub.file_download", "kokoro", "root"):
        logging.getLogger(name).addFilter(hide)
    for handler in logging.getLogger().handlers:
        handler.addFilter(hide)
