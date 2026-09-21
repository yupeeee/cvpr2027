"""One tqdm display, fed by nonblocking per-rank progress snapshots."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

from tqdm.auto import tqdm

from .config import DEFAULTS

_REPORTER = None


def _setting(args, name):
    return getattr(args, name, DEFAULTS[name])


class Reporter:
    """Workers publish small atomic files; only rank zero renders a single bar.

    No collectives occur during iteration. Completed phases remain available so
    a fast rank can move on while rank zero displays a slower rank's progress.
    """
    def __init__(self, args, dist, directory=None):
        self.args, self.dist = args, dist
        self.directory = Path(directory) if directory is not None else None
        self.interval = max(float(_setting(args, 'progress_mininterval')), 0.1)
        self.phases, self.stack, self.occurrences = {}, [], {}
        self.last_publish = 0.0
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.thread = None
        self.bar, self.display_key, self.error = None, None, None
        self.message = ''
        self.done = False

    def open(self, bar, aggregate):
        with self.lock:
            pinned = any(getattr(parent, 'overall', False) for parent in self.stack)
            if not pinned and (bar.position == 0 or not self.stack):
                occurrence = self.occurrences.get(bar.desc, 0)
                self.occurrences[bar.desc] = occurrence + 1
                bar.key = f'{bar.desc}#{occurrence}'
                self.phases[bar.key] = dict(n=bar.n, initial=bar.initial, total=bar.total,
                    desc=bar.desc, unit=bar.unit, started=time.time(), finished=False,
                    overall=bar.overall,
                    workers=self.dist.world_size if aggregate else 1)
            else:
                bar.key = None
            self.stack.append(bar)
            self.message = ''
            self.publish(force=True)
            if self.dist.is_main and self.thread is None:
                self.thread = threading.Thread(target=self._monitor, daemon=True)
                self.thread.start()

    def snapshot(self):
        detail = self.message
        if self.stack and not detail:
            current = self.stack[-1]
            detail = f'{current.desc} {current.n}/{current.total if current.total is not None else "?"}'
            if current.postfix:
                detail += ' ' + current.postfix
        return dict(rank=self.dist.rank, device=str(self.dist.device), phases=self.phases,
                    detail=detail, done=self.done)

    def publish(self, force=False):
        with self.lock:
            if self.error is not None:
                raise RuntimeError('Progress renderer failed') from self.error
            for bar in self.stack:
                if bar.key is not None:
                    self.phases[bar.key].update(n=bar.n, initial=bar.initial, total=bar.total, desc=bar.desc)
            if not force and time.monotonic() - self.last_publish < self.interval:
                return
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                target = self.directory / f'rank{self.dist.rank}.json'
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps(self.snapshot()), encoding='utf-8')
                os.replace(temporary, target)
            self.last_publish = time.monotonic()

    def close_bar(self, bar):
        with self.lock:
            if bar not in self.stack:
                return
            if bar.key is not None:
                self.phases[bar.key].update(n=bar.n, total=bar.total, desc=bar.desc,
                                            finished=True)
            self.stack.remove(bar)
            self.publish(force=True)

    def _snapshots(self):
        if self.directory is None:
            with self.lock:
                return [json.loads(json.dumps(self.snapshot()))]
        values = []
        for rank in range(self.dist.world_size):
            try:
                values.append(json.loads((self.directory / f'rank{rank}.json').read_text()))
            except FileNotFoundError:
                pass  # A worker may still be initializing; never wait for it here.
        return values

    def render(self):
        snapshots = self._snapshots()
        grouped = {}
        for value in snapshots:
            for key, phase in value['phases'].items():
                grouped.setdefault(key, []).append(phase)
        if not grouped:
            return
        pending = [(key, phases) for key, phases in grouped.items()
                   if len(phases) < max(p['workers'] for p in phases)
                   or not all(p['finished'] for p in phases)]
        if pending:
            key, phases = min(pending, key=lambda item: min(p['started'] for p in item[1]))
        else:
            key, phases = max(grouped.items(), key=lambda item: min(p['started'] for p in item[1]))
        ready = len(phases) == max(p['workers'] for p in phases)
        total = sum(p['total'] for p in phases) if ready and all(p['total'] is not None for p in phases) else None
        count, initial = sum(p['n'] for p in phases), sum(p['initial'] for p in phases)
        if self.bar is None:
            self.bar = tqdm(total=total, file=sys.stderr, dynamic_ncols=True,
                            mininterval=self.interval, miniters=1,
                            bar_format='[{elapsed}<{remaining}] {percentage:3.0f}%|{bar:8}| {desc} {n_fmt}/{total_fmt} {postfix}')
        if key != self.display_key:
            # reset() refreshes immediately; discard the old resume offset first.
            self.bar.initial = 0
            self.bar.set_description(phases[0]['desc'], refresh=False)
            self.bar.set_postfix_str('', refresh=False)
            self.bar.reset(total=total)
            self.bar.initial = initial
            self.bar.n = self.bar.last_print_n = initial
            self.bar.start_t = self.bar.last_print_t = min(p['started'] for p in phases)
            self.display_key = key
        elif initial != self.bar.initial:
            # A late worker may contribute cached work; do not count it as new throughput.
            cached_added = initial - self.bar.initial
            self.bar.n += cached_added
            self.bar.last_print_n += cached_added
            self.bar.initial = initial
        self.bar.bar_format = (
            '[{elapsed}<estimated {remaining}] {percentage:3.0f}%|{bar:8}| {desc} {n_fmt}/{total_fmt} {postfix}'
            if phases[0].get('overall') else
            '[{elapsed}<{remaining}] {percentage:3.0f}%|{bar:8}| {desc} {n_fmt}/{total_fmt} {postfix}')
        self.bar.total = total
        self.bar.unit = phases[0]['unit']
        by_rank = {value['rank']: value for value in snapshots}
        counts, details = [], []
        for rank in range(self.dist.world_size):
            value = by_rank.get(rank)
            if value is None:
                counts.append(f'rank{rank}: starting')
                continue
            phase = value['phases'].get(key)
            count_text = f"{phase['n']}/{phase['total']}" if phase else 'starting'
            label = f"{value['device']}/r{rank}" if self.dist.world_size > 1 else value['device']
            counts.append(f"{label} {count_text}")
            details.append(f"{label}: {value['detail']}")
        self.bar.set_description(" | ".join(counts) + " | " + phases[0]["desc"], refresh=False)
        self.bar.set_postfix_str(' | '.join(details), refresh=False)
        self.bar.update(count - self.bar.n)
        self.bar.refresh()

    def _monitor(self):
        try:
            while not self.stopped.is_set():
                self.render()
                self.stopped.wait(self.interval)
        except Exception as exc:
            self.error = exc

    def close(self, error=None):
        self.done = True
        self.message = f'failed: {error}' if error else 'finished'
        self.publish(force=True)
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.render()
        if self.bar is not None:
            self.bar.close()


class _ReportedProgress:
    """Small tqdm-compatible counter; all presentation belongs to Reporter."""
    def __init__(self, reporter, iterable, desc, total, initial, unit, position, aggregate, overall=False):
        self.reporter, self.iterable = reporter, iterable
        self.desc, self.unit, self.position = desc, unit, position
        self.overall = overall
        self.total = total if total is not None else len(iterable) if hasattr(iterable, '__len__') else None
        self.n = self.initial = initial
        self.postfix = ''
        reporter.open(self, aggregate)

    def update(self, n=1):
        self.reporter.message = ''
        self.n += n
        self.reporter.publish()

    def set_description(self, desc, refresh=True):
        self.desc = desc
        if refresh:
            self.refresh()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        self.postfix = ', '.join(f'{key}={value}' for key, value in {**(ordered_dict or {}), **kwargs}.items())
        if refresh:
            self.refresh()

    def set_postfix_str(self, value, refresh=True):
        self.postfix = value
        if refresh:
            self.refresh()

    def refresh(self):
        self.reporter.publish()

    def close(self):
        self.reporter.close_bar(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        try:
            for item in self.iterable:
                yield item
                self.update()
        finally:
            self.close()


def start_progress(args, dist):
    """One setup rendezvous; no CUDA work or progress collectives in trial loops."""
    global _REPORTER
    if not _setting(args, 'progress'):
        return
    directory = None
    if dist.world_size > 1:
        import torch.distributed as distributed
        session = [uuid.uuid4().hex if dist.is_main else None]
        distributed.broadcast_object_list(session, src=0, device=dist.device)
        directory = Path(args.logs_dir) / args.run_id / 'progress' / session[0]
    _REPORTER = Reporter(args, dist, directory)


def stop_progress(error=None):
    global _REPORTER
    reporter, _REPORTER = _REPORTER, None
    if reporter is not None:
        reporter.close(error)


def progress(args, iterable=None, *, desc, total=None, initial=0, unit='it',
             dist=None, position=0, leave=True, overall=False):
    """Report aggregated work. ``overall`` pins a planned total across nested phases.

    Nested bars then publish operation details only: their positions cannot reset
    the outer count or ETA. The caller advances the outer weighted-work counter.
    """
    enabled = _setting(args, 'progress') and not getattr(args, 'dry_run', False)
    if enabled and _REPORTER is not None:
        return _ReportedProgress(_REPORTER, iterable, desc, total, initial, unit,
                                 position, aggregate=dist is not None, overall=overall)
    rank = getattr(dist, 'rank', int(os.environ.get('RANK', '0')))
    return tqdm(iterable, desc=desc, total=total, initial=initial, unit=unit,
                disable=not enabled or rank != 0, file=sys.stderr,
                mininterval=_setting(args, 'progress_mininterval'), miniters=1,
                dynamic_ncols=True, position=position, leave=leave)


def status(args, message, dist=None):
    if _setting(args, 'progress') and not getattr(args, 'dry_run', False):
        if _REPORTER is not None:
            _REPORTER.message = message
            _REPORTER.publish(force=True)
        elif getattr(dist, 'rank', int(os.environ.get('RANK', '0'))) == 0:
            tqdm.write(message, file=sys.stderr)


def quiet_library_progress():
    """Keep model-library bars from competing with the single pilot display."""
    from diffusers.utils.logging import disable_progress_bar as disable_diffusers
    from transformers.utils.logging import disable_progress_bar as disable_transformers
    from huggingface_hub.utils import disable_progress_bars
    disable_diffusers()
    disable_transformers()
    disable_progress_bars()
