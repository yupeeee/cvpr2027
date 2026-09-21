"""Small local files exercise receipt reuse without model or tensor loading."""
import os

import pytest

from utils import validation
from utils.validation import ArtifactValidationCache


def fixture(tmp_path, identity=None):
    path = tmp_path / 'artifact.bin'
    path.write_bytes(b'valid artifact')
    directory = tmp_path / 'validation'
    return path, directory, ArtifactValidationCache(directory, identity or {'schema': 1})


def forbidden():
    pytest.fail('Unchanged valid artifact was read or validated again')


def test_unchanged_success_persists_without_hash_or_validation(tmp_path, monkeypatch):
    path, directory, cache = fixture(tmp_path)
    assert cache.check(['trajectory', 0], [path], lambda: True)
    cache.flush()
    monkeypatch.setattr(validation, '_sha256', lambda *a: forbidden())
    restarted = ArtifactValidationCache(directory, {'schema': 1})
    assert restarted.check(['trajectory', 0], [path], forbidden)


def test_touched_identical_content_hashes_without_deserializing(tmp_path, monkeypatch):
    path, _, cache = fixture(tmp_path)
    assert cache.check('key', [path], lambda: True)
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    original = validation._sha256
    hashed = []
    monkeypatch.setattr(validation, '_sha256', lambda *args: hashed.append(1) or original(*args))
    assert cache.check('key', [path], forbidden)
    assert len(hashed) == 1
    assert cache.check('key', [path], forbidden)
    assert len(hashed) == 1


def test_changed_bytes_validate_and_failed_results_are_not_certified(tmp_path):
    path, directory, cache = fixture(tmp_path)
    assert cache.check('key', [path], lambda: True)
    path.write_bytes(b'changed artifact')
    seen = []
    assert not cache.check('key', [path], lambda: seen.append(1) or False)
    assert not list(directory.glob('*.json'))
    assert cache.check('key', [path], lambda: seen.append(1) or True)
    assert len(seen) == 2


def test_missing_or_non_file_artifact_invalidates_receipt(tmp_path):
    path, directory, cache = fixture(tmp_path)
    assert cache.check('key', [path], lambda: True)
    path.unlink()
    assert not cache.check('key', [path], forbidden)
    assert not list(directory.glob('*.json'))
    path.mkdir()
    assert not cache.check('key', [path], forbidden)


@pytest.mark.parametrize('broken', ['not json', '{"version": 1}', '{"version": 1, "token": 2, "files": null}'])
def test_corrupt_receipt_falls_back_to_validation(tmp_path, broken):
    path, directory, cache = fixture(tmp_path)
    assert cache.check('key', [path], lambda: True)
    next(directory.glob('*.json')).write_text(broken)
    seen = []
    assert cache.check('key', [path], lambda: seen.append(1) or True)
    assert seen == [1]


def test_identity_and_logical_key_are_part_of_receipt(tmp_path):
    path, directory, cache = fixture(tmp_path)
    assert cache.check('key', [path], lambda: True)
    seen = []
    changed = ArtifactValidationCache(directory, {'schema': 2})
    assert changed.check('key', [path], lambda: seen.append(1) or True)
    assert cache.check('different-key', [path], lambda: seen.append(1) or True)
    assert seen == [1, 1]


def test_hash_time_modification_does_not_certify(tmp_path, monkeypatch):
    path, directory, cache = fixture(tmp_path)
    original = validation._sha256

    def modify(source, info):
        digest = original(source, info)
        source.write_bytes(b'changed during hashing')
        return digest

    monkeypatch.setattr(validation, '_sha256', modify)
    assert not cache.check('key', [path], forbidden)
    assert not list(directory.glob('*.json'))


def test_validation_time_modification_does_not_certify(tmp_path):
    path, directory, cache = fixture(tmp_path)

    def modify():
        path.write_bytes(b'changed during strict validation')
        return True

    assert not cache.check('key', [path], modify)
    assert not list(directory.glob('*.json'))


def test_validator_exception_discards_previous_receipt(tmp_path):
    path, directory, cache = fixture(tmp_path)
    assert cache.check('key', [path], lambda: True)
    path.write_bytes(b'changed artifact')

    def invalid():
        raise ValueError('Strict validation failed')

    with pytest.raises(ValueError, match='Strict validation failed'):
        cache.check('key', [path], invalid)
    assert not list(directory.glob('*.json'))


def test_all_dependency_files_must_match(tmp_path):
    path, _, cache = fixture(tmp_path)
    dependency = tmp_path / 'metadata.json'
    dependency.write_text('{}')
    assert cache.check('key', [path, dependency], lambda: True)
    dependency.write_text('{"changed": true}')
    assert not cache.check('key', [path, dependency], lambda: False)


def test_unwritable_receipt_directory_does_not_invalidate_good_artifacts(tmp_path, monkeypatch):
    path, _, cache = fixture(tmp_path)

    def fail_write(*args):
        raise PermissionError('receipt directory is read only')

    monkeypatch.setattr(validation, 'atomic_json', fail_write)
    assert cache.check('key', [path], lambda: True)
