"""Role execution support: bounded calls, successful-result cache and archives.

This wrapper is installed by the judge's caller. Backends retain ownership
of HTTP timeouts/retries and response/model validation. No workers or clients
are created at import time. Cache ownership is one runtime instance, bounded by
``cache_size``; request and source identities are rebound on every cache hit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from decision_model.types import DecisionResult


class JsonlArchive:
    """Explicit opt-in local archive; never records headers or model options."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, record):
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")


class EvaluationRuntime:
    """A complete/complete_many adapter for role-owned execution support.

    ``cache_identity`` supplies extra result-affecting model settings for custom
    backends. For encodable backends, the full encoded request is the cache key;
    otherwise full context, question and request metadata are included. Local
    request/source IDs are intentionally excluded. Configured model/version and
    endpoint are included independently of the encoded request.
    """

    def __init__(self, model, *, max_workers=4, cache_size=0, timeout=None,
                 cache_identity=None, archive=None):
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        if timeout is not None and (isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be positive and finite")
        self.model = model
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.timeout = timeout
        self.cache_identity = copy.deepcopy(cache_identity or {})
        self.archive = JsonlArchive(archive) if isinstance(archive, (str, Path)) else archive
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._stats = dict(requests=0, calls=0, cache_hits=0, successes=0,
                           failures=0, retries=0, elapsed_s=0.0, cost=0.0)

    def _key(self, request):
        encoded = getattr(self.model, "encode", None)
        payload = encoded(request) if callable(encoded) else {
            "context": request.context.to_dict(), "question": request.question.to_dict(),
            "meta": dict(request.meta),
        }
        identity = {name: getattr(self.model, name, None)
                    for name in ("name", "model_name", "expected_model")}
        endpoint = getattr(self.model, "endpoint", None)
        if endpoint is None:
            # Third-party backends may still expose only a legacy base address.
            endpoint = getattr(self.model, "api_base", None)
            if endpoint is None:
                endpoint = getattr(self.model, "base_url", None)
        identity["endpoint"] = endpoint
        data = {"model": identity, "options": self.cache_identity, "request": payload}
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False).encode("utf-8")).hexdigest()

    def complete(self, request, *, timeout=None):
        started = time.monotonic()
        cache_hit = False
        try:
            key = self._key(request) if self.cache_size else None
        except Exception:
            # Let the backend turn an unencodable request into its normal typed
            # failure. A cache cannot make a request bypass failure policy.
            key = None
        with self._lock:
            cached = self._cache.get(key) if key is not None else None
            if cached is not None:
                self._cache.move_to_end(key)
        if cached is not None:
            result = replace(copy.deepcopy(cached), request_id=request.request_id)
            cache_hit = True
        else:
            try:
                result = self.model.complete(request, timeout=self.timeout if timeout is None else timeout)
            except Exception as exc:
                # A backend exception may contain credentials; archive only its type.
                result = DecisionResult.failure(request.request_id, "backend_exception:" + type(exc).__name__)
            if not isinstance(result, DecisionResult):
                result = DecisionResult.failure(request.request_id, "invalid_result_type")
            elif result.request_id != request.request_id:
                result = DecisionResult.failure(request.request_id, "mismatched_request_id")
            if key is not None and result.is_ok:
                with self._lock:
                    self._cache[key] = copy.deepcopy(result)
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        elapsed = time.monotonic() - started
        with self._lock:
            self._stats["requests"] += 1
            self._stats["cache_hits"] += int(cache_hit)
            self._stats["calls"] += int(not cache_hit)
            self._stats["successes" if result.is_ok else "failures"] += 1
            self._stats["elapsed_s"] += elapsed
            if not cache_hit:
                attempts = result.raw.get("attempts", 1)
                self._stats["retries"] += max(0, attempts - 1) if type(attempts) is int else 0
                # Public backends may expose usage directly or retain a native
                # response envelope. Count one source, without provider routing.
                usage = result.raw.get("usage")
                if not isinstance(usage, Mapping):
                    response = result.raw.get("response")
                    usage = response.get("usage") if isinstance(response, Mapping) else None
                cost = usage.get("cost", 0) if isinstance(usage, Mapping) else 0
                if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost):
                    self._stats["cost"] += cost
        if self.archive is not None:
            self.archive({"request": request.to_dict(), "result": result.to_dict(),
                          "cache_hit": cache_hit, "elapsed_s": elapsed})
        return result

    def complete_many(self, requests, *, timeout=None):
        # executor.map retains input identity/order while at most max_workers calls
        # run. No backend complete_many implementation is needed for concurrency.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(lambda request: self.complete(request, timeout=timeout), requests))

    def statistics(self):
        with self._lock:
            return dict(self._stats)

    def clear_cache(self):
        with self._lock:
            self._cache.clear()

    def close(self):
        self.clear_cache()
        self.model.close()
