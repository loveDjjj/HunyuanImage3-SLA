"""Context propagated through activation checkpoint recomputation."""

from __future__ import annotations

import contextlib
import contextvars


_FULL_ATTN_SPANS = contextvars.ContextVar("hunyuan_sla_full_attention_spans", default=None)


def current_sla_full_attention_spans():
    return _FULL_ATTN_SPANS.get()


@contextlib.contextmanager
def sla_full_attention_spans(spans):
    token = _FULL_ATTN_SPANS.set(spans)
    try:
        yield
    finally:
        _FULL_ATTN_SPANS.reset(token)
