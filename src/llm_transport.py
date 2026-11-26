"""Retry transient local inference outages in supervised runs."""
import os
import time
import requests


def post_generation(resolve_url, payload, timeout):
    persistent = os.getenv('LLMGE_PERSISTENT') == '1'
    while True:
        try:
            # Resolve every attempt: replacement Slurm servers can move to another node.
            response = requests.post(resolve_url(), headers={'Content-Type': 'application/json'},
                                     json=payload, timeout=(10, timeout))
            if not persistent or response.status_code not in (408, 429, 502, 503, 504):
                return response
            reason = f'HTTP {response.status_code}'
            response.close()
        except (requests.exceptions.RequestException, OSError, ValueError) as error:
            if not persistent:
                raise
            reason = str(error)
        print(f'LLM unavailable ({reason}); preserving request and retrying in 30s', flush=True)
        time.sleep(30)
