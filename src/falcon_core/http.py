"""
Resilient HTTP client with incremental backoff.

All API requests — internal and external — should go through this module
so failures are visible and retries are consistent.

Internal (container-to-container): starts at 50ms, caps at 2s
External (Polygon, Finviz, Yahoo): starts at 500ms, caps at 30s

Usage:
    from falcon_core.http import http_get, http_post

    # External API call with defaults
    resp = http_get("https://api.polygon.io/v2/...", params={...})

    # Internal call with faster backoff
    resp = http_get("http://falcon-dashboard:5000/api/market", profile="internal")

    # Custom config
    resp = http_get(url, max_retries=5, initial_delay_ms=100)
"""

import time
import logging
import random
from typing import Any, Dict, Optional

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

# Backoff profiles
PROFILES = {
    "internal": {
        "max_retries": 3,
        "initial_delay_ms": 50,
        "max_delay_ms": 2000,
        "backoff_factor": 2.0,
        "jitter": True,
    },
    "external": {
        "max_retries": 3,
        "initial_delay_ms": 500,
        "max_delay_ms": 30000,
        "backoff_factor": 2.0,
        "jitter": True,
    },
}

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetryExhausted(Exception):
    """All retry attempts failed."""
    def __init__(self, url: str, attempts: int, last_error: str):
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Failed after {attempts} attempts: {url} — {last_error}")


def _calculate_delay(attempt: int, initial_ms: float, max_ms: float,
                     factor: float, jitter: bool,
                     retry_after: Optional[float] = None) -> float:
    """Calculate backoff delay in seconds."""
    if retry_after is not None:
        # Server told us how long to wait — respect it
        return min(retry_after, max_ms / 1000)

    delay_ms = initial_ms * (factor ** attempt)
    delay_ms = min(delay_ms, max_ms)

    if jitter:
        # Add up to 25% jitter to prevent thundering herd
        delay_ms *= (1.0 + random.uniform(0, 0.25))

    return delay_ms / 1000  # convert to seconds


def _parse_retry_after(response: requests.Response) -> Optional[float]:
    """Parse Retry-After header (seconds or HTTP-date)."""
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _do_request(method: str, url: str,
                max_retries: int = None,
                initial_delay_ms: float = None,
                max_delay_ms: float = None,
                backoff_factor: float = None,
                jitter: bool = None,
                profile: str = "external",
                timeout: float = 10,
                raise_on_failure: bool = False,
                **kwargs) -> Optional[requests.Response]:
    """
    Make an HTTP request with incremental backoff.

    Args:
        method: "GET" or "POST"
        url: Request URL
        max_retries: Override max retry count
        initial_delay_ms: Override initial backoff (milliseconds)
        max_delay_ms: Override max backoff (milliseconds)
        backoff_factor: Override backoff multiplier
        jitter: Override jitter setting
        profile: "internal" or "external" (sets defaults)
        timeout: Request timeout in seconds
        raise_on_failure: If True, raise RetryExhausted instead of returning None
        **kwargs: Passed to requests.get/post (params, headers, json, cookies, etc.)

    Returns:
        requests.Response or None if all retries exhausted (unless raise_on_failure)
    """
    # Merge profile defaults with explicit overrides
    defaults = PROFILES.get(profile, PROFILES["external"])
    cfg = {
        "max_retries": max_retries if max_retries is not None else defaults["max_retries"],
        "initial_delay_ms": initial_delay_ms if initial_delay_ms is not None else defaults["initial_delay_ms"],
        "max_delay_ms": max_delay_ms if max_delay_ms is not None else defaults["max_delay_ms"],
        "backoff_factor": backoff_factor if backoff_factor is not None else defaults["backoff_factor"],
        "jitter": jitter if jitter is not None else defaults["jitter"],
    }

    last_error = ""
    request_fn = requests.get if method.upper() == "GET" else requests.post

    for attempt in range(cfg["max_retries"] + 1):
        try:
            response = request_fn(url, timeout=timeout, **kwargs)

            # Success — return immediately
            if response.status_code not in RETRYABLE_STATUS_CODES:
                if attempt > 0:
                    logger.info("Request succeeded on attempt %d: %s %s",
                                attempt + 1, method, url)
                return response

            # Retryable status code
            retry_after = _parse_retry_after(response)
            last_error = f"HTTP {response.status_code}"

            if response.status_code == 429:
                last_error = f"HTTP 429 (rate limited)"
                logger.warning("Rate limited on %s %s (attempt %d/%d)",
                               method, url, attempt + 1, cfg["max_retries"] + 1)
            else:
                logger.warning("Retryable HTTP %d on %s %s (attempt %d/%d)",
                               response.status_code, method, url,
                               attempt + 1, cfg["max_retries"] + 1)

            # Last attempt — return the error response rather than retrying
            if attempt >= cfg["max_retries"]:
                break

            delay = _calculate_delay(
                attempt, cfg["initial_delay_ms"], cfg["max_delay_ms"],
                cfg["backoff_factor"], cfg["jitter"], retry_after,
            )
            logger.info("Backing off %.1fs before retry %d: %s",
                        delay, attempt + 2, url)
            time.sleep(delay)

        except RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.warning("Request error on %s %s (attempt %d/%d): %s",
                           method, url, attempt + 1, cfg["max_retries"] + 1, last_error)

            if attempt >= cfg["max_retries"]:
                break

            delay = _calculate_delay(
                attempt, cfg["initial_delay_ms"], cfg["max_delay_ms"],
                cfg["backoff_factor"], cfg["jitter"],
            )
            logger.info("Backing off %.1fs before retry %d: %s",
                        delay, attempt + 2, url)
            time.sleep(delay)

    # All retries exhausted
    logger.error("All %d attempts failed for %s %s: %s",
                 cfg["max_retries"] + 1, method, url, last_error)

    if raise_on_failure:
        raise RetryExhausted(url, cfg["max_retries"] + 1, last_error)
    return None


def http_get(url: str, **kwargs) -> Optional[requests.Response]:
    """GET with incremental backoff. See _do_request for args."""
    return _do_request("GET", url, **kwargs)


def http_post(url: str, **kwargs) -> Optional[requests.Response]:
    """POST with incremental backoff. See _do_request for args."""
    return _do_request("POST", url, **kwargs)
