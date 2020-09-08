import tempfile
from contextlib import contextmanager


@contextmanager
def get_storage_url(specifier):
    tmpfile = None
    try:
        if specifier == "inmemory":
            url = None
        elif specifier == "sqlite":
            tmpfile = tempfile.NamedTemporaryFile()
            url = "sqlite:///{}".format(tmpfile.name)
        else:
            raise ValueError(
                "Invalid specifier entered. Was expecting 'inmemory' or 'sqlite'"
                f"but got {specifier} instead"
            )
        yield url
    finally:
        if tmpfile is not None:
            tmpfile.close()
