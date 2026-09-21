"""Successful local-artifact checks, reused while file identity stays unchanged.

SHA256 is calculated once on first validation and whenever filesystem metadata
changes. An unchanged artifact needs only stat calls and its small JSON receipt;
this avoids repeatedly reading/deserializing large tensors on network storage.
Receipts are an optional accelerator, never a substitute for initial validation.
"""
import hashlib
import json
import os
import stat
from pathlib import Path

from .data import atomic_json


_VERSION = 1


def _metadata(info):
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev]


def _snapshot(paths):
    records = []
    for path in paths:
        resolved = str(path.resolve(strict=True))
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise OSError(f'Expected a regular artifact file: {path}')
        records.append({'path': resolved, 'stat': _metadata(info)})
    return records


def _sha256(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        if _metadata(os.fstat(source.fileno())) != expected['stat']:
            raise OSError(f'Artifact changed before hashing: {path}')
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
        if _metadata(os.fstat(source.fileno())) != expected['stat']:
            raise OSError(f'Artifact changed while hashing: {path}')
    return digest.hexdigest()


class ArtifactValidationCache:
    """Persist positive checks in one atomic JSON file per logical key.

    ``identity`` must include the scientific/component identity and the caller's
    validation schema version. ``paths`` must contain every file that the strict
    ``validate()`` callback depends on. Use the logical key for other dependencies.
    Independent ranks can safely replace the same receipt; stale receipts fail
    the next metadata comparison. Cache I/O failure merely disables acceleration.
    """

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.identity = identity

    def _location(self, key):
        payload = json.dumps([_VERSION, self.identity, key], sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode()
        token = hashlib.sha256(payload).hexdigest()
        return self.directory / f'{token}.json', token

    @staticmethod
    def _discard(path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass  # Receipts are optional and a stale entry cannot pass new stats.

    @staticmethod
    def _read(path, token):
        try:
            receipt = json.loads(path.read_text())
            if receipt['version'] != _VERSION or receipt['token'] != token:
                return None
            files = receipt['files']
            if not isinstance(files, list):
                return None
            for item in files:
                if (not isinstance(item['path'], str) or
                        not isinstance(item['stat'], list) or len(item['stat']) != 5 or
                        any(type(value) is not int for value in item['stat']) or
                        not isinstance(item['sha256'], str) or len(item['sha256']) != 64 or
                        any(c not in '0123456789abcdef' for c in item['sha256'])):
                    return None
            return files
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def check(self, key, paths, validate):
        """Return strict validity, reusing a successful unchanged artifact check.

        Missing/unreadable/changing files return False. Validator exceptions are
        propagated after discarding the old receipt; they are never certified.
        """
        paths = [Path(path) for path in paths]
        receipt_path, token = self._location(key)
        previous = self._read(receipt_path, token)
        try:
            before = _snapshot(paths)
            if previous is not None and before == [
                    {'path': item['path'], 'stat': item['stat']} for item in previous]:
                return True
            digests = [_sha256(path, info) for path, info in zip(paths, before)]
            if _snapshot(paths) != before:
                self._discard(receipt_path)
                return False
        except OSError:
            self._discard(receipt_path)
            return False
        same_bytes = previous is not None and [
            (item['path'], item['sha256']) for item in previous
        ] == [(item['path'], digest) for item, digest in zip(before, digests)]
        if not same_bytes:
            self._discard(receipt_path)
            if not validate():
                return False
        try:
            if _snapshot(paths) != before:
                self._discard(receipt_path)
                return False
            files = [dict(info, sha256=digest) for info, digest in zip(before, digests)]
            atomic_json(receipt_path, {'version': _VERSION, 'token': token, 'files': files})
        except OSError:
            # Distinguish artifact races from a cache directory we cannot write.
            try:
                if _snapshot(paths) != before:
                    self._discard(receipt_path)
                    return False
            except OSError:
                self._discard(receipt_path)
                return False
        return True

    def flush(self):
        """No buffering: successful checks are already saved atomically."""
