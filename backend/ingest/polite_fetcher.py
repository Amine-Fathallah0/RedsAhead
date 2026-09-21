import hashlib
import time
from pathlib import Path
import requests

class PoliteFetcher:
    def __init__(
        self,
        cache_dir,
        min_interval_seconds,
        session=None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_seconds = min_interval_seconds
        self.session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request = None

    def get(self, url: str) -> str:
        name=hashlib.sha256(url.encode("utf-8")).hexdigest() +".html"
        path=Path(self.cache_dir) / name
        if path.exists():
            return path.read_text(encoding="utf-8")
        self._wait_if_needed()
        response = self.session.get(url, timeout=30)
        self._last_request = self._clock()
        response.raise_for_status()
        text = response.text
        path.write_text(text, encoding="utf-8")
        return text

    def _wait_if_needed(self) -> None:
        if self._last_request is None:
            return
        wait = self._last_request + self.min_interval_seconds - self._clock()
        if wait > 0:
            self._sleep(wait)

        
        

