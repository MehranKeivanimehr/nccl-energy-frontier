"""Per-test warning discipline for the Stage 2 tests (no global filter is installed or changed).

@strict: every warning raised inside the test is an error (unexpected numerical warnings fail the test).
expect_warning(category, match): a warning the test exercises on purpose is captured locally and asserted.
"""
import contextlib
import functools
import re
import warnings


def strict(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return fn(*a, **k)
    return wrapper


@contextlib.contextmanager
def expect_warning(category, match: str):
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        yield got
    hits = [w for w in got if issubclass(w.category, category) and re.search(match, str(w.message))]
    others = [w for w in got if w not in hits]
    assert hits, f"expected {category.__name__} matching {match!r}; got {[str(w.message) for w in got]}"
    assert not others, f"unexpected warnings: {[(w.category.__name__, str(w.message)) for w in others]}"
