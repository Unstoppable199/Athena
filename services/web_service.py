"""
Web capability.

Searches the web, extracts page content, and ranks it by
relevance to the search query before returning it.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from urllib.parse import urlsplit

from ddgs import DDGS
import requests
from bs4 import BeautifulSoup


class WebService:

    MAX_CONTENT_LENGTH = 3000
    MAX_RAW_LENGTH = 50000
    MAX_DOWNLOAD_BYTES = 750000
    CONTEXT_WINDOW = 1
    CACHE_TTL_SECONDS = 600  # 10 minutes

    # "general" domains are considered trustworthy for ANY category.
    # Category-specific domains only count as trusted for that
    # category's searches, on top of the general list.
    GENERAL_DOMAINS = {
        # Reference / encyclopedic
        "wikipedia.org", "britannica.com",

        # Wire services & major international news
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
        "cnn.com", "nytimes.com", "theguardian.com", "npr.org",
        "aljazeera.com", "washingtonpost.com",

        # Indian news (major national outlets)
        "ndtv.com", "thehindu.com", "hindustantimes.com",
        "indianexpress.com", "timesofindia.indiatimes.com",

        # Government / official / international bodies
        "whitehouse.gov", ".gov", ".europa.eu", "un.org", "who.int",
    }

    FINANCE_DOMAINS = {
        "bloomberg.com", "ft.com", "wsj.com", "economist.com",
        "imf.org", "worldbank.org",
    }

    SPORTS_DOMAINS = {
        "espn.com", "skysports.com", "olympics.com",
        "fifa.com", "uefa.com", "nba.com", "nfl.com", "fifa.org",
        "cricbuzz.com", "espncricinfo.com",
        "sportingnews.com",
    }

    WEATHER_DOMAINS = {
        "weather.com", "wunderground.com",
    }

    CATEGORY_DOMAINS = {
        "finance": FINANCE_DOMAINS,
        "sports": SPORTS_DOMAINS,
        "weather": WEATHER_DOMAINS,
        "general": set(),
    }

    # Sites observed to consistently block scraping (403/406) or be
    # JS-rendered live-score apps that return nothing via requests+bs4.
    # Skipping them outright saves the wasted round-trip/timeout.
    SKIP_DOMAINS = {
        "sofascore.com", "besoccer.com", "goal.com", "goals365.live",
        "timeanddate.com", "accuweather.com",
    }

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "in", "on",
        "at", "of", "for", "to", "and", "or", "what", "whats",
        "current", "right", "now", "today", "latest", "please", "me", "i"
    }

    # Words that describe the kind of lookup rather than its subject.
    # Removing these leaves the product, organisation or place whose
    # own domain can be recognised as an official source.  For example,
    # "latest stable Python version" leaves "python", which matches
    # python.org; a tutorial site merely containing the word Python in
    # its path does not.
    AUTHORITY_STOPWORDS = STOPWORDS | {
        "last", "most", "recent", "stable", "version", "versions",
        "release", "releases", "download", "downloads", "official",
        "winner", "winners", "result", "results", "score", "price",
        "weather", "forecast", "news", "update", "updates",
    }

    USER_AGENT = (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/137.0 Safari/537.36"
    )

    def __init__(self):
        self._cache = {}

    def _cache_key(self, query: str, category: str) -> str:
        return f"{query.strip().lower()}|{category}"

    def _is_trusted(self, url: str, category: str = "general") -> bool:

        if not url:
            return False

        domains = self.GENERAL_DOMAINS | self.CATEGORY_DOMAINS.get(category, set())

        try:
            host = (urlsplit(url).hostname or "").casefold().strip(".")
        except ValueError:
            return False

        def matches(domain: str) -> bool:
            domain = domain.casefold()

            # Entries beginning with a dot intentionally mean any host
            # under that public suffix (for example any .gov site).
            if domain.startswith("."):
                return host.endswith(domain)

            return host == domain or host.endswith("." + domain)

        return any(matches(domain) for domain in domains)

    @staticmethod
    def _site_label(url: str) -> str:
        """Best-effort owner label from a normal public hostname.

        No network/public-suffix lookup is needed.  The small ccTLD
        exception handles common hosts such as example.co.uk; known
        news and government domains are already covered by the curated
        trust sets above.
        """

        try:
            host = (urlsplit(url).hostname or "").casefold().strip(".")
        except ValueError:
            return ""

        parts = [part for part in host.split(".") if part]
        if len(parts) < 2:
            return ""

        label_index = -2
        if (
            len(parts) >= 3
            and len(parts[-1]) == 2
            and parts[-2] in {"ac", "co", "com", "edu", "gov", "net", "org"}
        ):
            label_index = -3

        return parts[label_index]

    def _is_official_for_query(self, url: str, query: str) -> bool:
        """Whether the subject appears to own the result's domain.

        This deliberately requires an exact owner-label match, not a
        substring anywhere in a URL.  It therefore recognises
        python.org for a Python query and fifa.com for a FIFA query,
        without promoting notpython.org or an article whose path merely
        mentions Python.
        """

        label = self._site_label(url)
        if not label:
            return False

        query_words = {
            word for word in re.findall(r"[a-z0-9]+", str(query).casefold())
            if (
                len(word) >= 3
                and not word.isdigit()
                and word not in self.AUTHORITY_STOPWORDS
            )
        }
        label_words = set(re.findall(r"[a-z0-9]+", label))

        return bool(query_words & (label_words | {label}))

    def search(self, query: str, category: str = "general", max_results: int = 4):

        cache_key = self._cache_key(query, category)
        cached = self._cache.get(cache_key)

        if cached:
            cached_at, cached_result = cached
            if time.time() - cached_at < self.CACHE_TTL_SECONDS:
                print(f"[WEB SEARCH CACHE HIT] {query!r} ({category})")
                return cached_result
            else:
                del self._cache[cache_key]

        try:

            with DDGS() as ddgs:

                search_results = list(
                    ddgs.text(
                        query,
                        max_results=max_results
                    )
                )

            urls = [
                result.get("href")
                for result in search_results
            ]

            with ThreadPoolExecutor(max_workers=5) as executor:

                contents = list(
                    executor.map(
                        partial(self.fetch_page, query=query, category=category),
                        urls
                    )
                )

            pages = []

            for result, content in zip(search_results, contents):

                snippet = result.get("body", "")

                if not content.strip() and not snippet.strip():
                    continue

                url = result.get("href", "")
                official = self._is_official_for_query(url, query)

                print(f"[WEB SEARCH] {url}")   # <-- add this line
                
                pages.append(
                    {
                        "source": "web",
                        "title": result.get("title", ""),
                        "url": url,
                        "snippet": snippet,
                        "content": content,
                        "official": official,
                        "trusted": official or self._is_trusted(url, category)
                    }
                )

            # The subject's own site is stronger than a generally
            # reputable secondary source.  Keep both, but put authority
            # first so a small local model sees the best evidence early.
            pages.sort(key=lambda p: (not p["official"], not p["trusted"]))

            result = {
                "success": True,
                "data": pages
            }

            self._cache[cache_key] = (time.time(), result)

            return result

        except Exception as e:

            return {
                "success": False,
                "error": str(e)
            }

    def fetch_page(self, url: str, query: str = "", category: str = "general") -> str:
        if not url:
            return ""

        try:
            host = (urlsplit(url).hostname or "").casefold()
        except ValueError:
            return ""

        if any(
            host == domain or host.endswith("." + domain)
            for domain in self.SKIP_DOMAINS
        ):
            return ""

        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.USER_AGENT},
                timeout=(3, 6),
                stream=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").casefold()
            if content_type and not any(
                kind in content_type for kind in ("text/", "html", "xml")
            ):
                return ""

            chunks = []
            downloaded = 0
            for chunk in response.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > self.MAX_DOWNLOAD_BYTES:
                    break
                chunks.append(chunk)

            body = b"".join(chunks)
            encoding = response.encoding or "utf-8"
            page_text = body.decode(encoding, errors="replace")

            soup = BeautifulSoup(page_text, "html.parser")

            strip_tags = [
                "script", "style", "noscript", "svg", "canvas", "iframe",
                "nav", "header", "footer", "aside",
            ]
            if category not in {"sports", "finance"}:
                strip_tags.append("table")

            for tag in soup(strip_tags):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)

            raw_text = "\n".join(lines)[:self.MAX_RAW_LENGTH]

            if not query:
                return raw_text[:self.MAX_CONTENT_LENGTH]

            return self._rank_by_relevance(raw_text, query)

        except Exception as e:
            print(f"[WEB FETCH ERROR] {url}: {e}")
            return ""

    def _tokenize(self, text: str):

        words = re.findall(r"[a-z0-9]+", text.lower())

        return [w for w in words if w not in self.STOPWORDS]

    def _rank_by_relevance(self, text: str, query: str) -> str:

        lines = text.split("\n")
        query_terms = set(self._tokenize(query))

        if not query_terms:
            return text[:self.MAX_CONTENT_LENGTH]

        scored = []
        line_scores = {}
        
        JUNK_PATTERNS = re.compile(
            r"^(jump to|edit|\[.*\]|photo|image from|see also|references"
            r"|from wikipedia|full \d+ ?hour|advertisement|cookie"
            r"|\"?[\w\s]+\"? and \"?[\w\s]+\"? redirect"
            r"|[a-z]+,\s[a-z]+\s\(\d{1,2}\s[a-z]+\s\d{4}\))",
            re.IGNORECASE
        )

        for i, line in enumerate(lines):
            if len(line) < 10 or JUNK_PATTERNS.match(line.strip()):
                continue
            line_terms = set(self._tokenize(line))
            overlap = query_terms.intersection(line_terms)
            score = len(overlap) * 2
            if any(ch.isdigit() for ch in line):
                score += 1
            scored.append((i, score))
            line_scores[i] = score

        scored.sort(key=lambda x: x[1], reverse=True)

        selected = set()

        for i, score in scored:

            if score == 0:
                continue

            for offset in range(-self.CONTEXT_WINDOW, self.CONTEXT_WINDOW + 1):
                idx = i + offset
                if (0 <= idx < len(lines)
                        and len(lines[idx].strip()) >= 10
                        and not JUNK_PATTERNS.match(lines[idx].strip())
                        # Nearby text is useful only when it contains a query
                        # term or a concrete value.  Previously every adjacent
                        # navigation/footer line was pulled in with a match.
                        and line_scores.get(idx, 0) > 0):
                    selected.add(idx)

            candidate = "\n".join(lines[j] for j in sorted(selected))

            if len(candidate) > self.MAX_CONTENT_LENGTH:
                break

        if not selected:
            return ""

        unique = []
        seen = set()
        for index in sorted(selected):
            line = lines[index].strip()
            key = re.sub(r"\W+", " ", line.casefold()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(line)

        result = "\n".join(unique)

        return result[:self.MAX_CONTENT_LENGTH]
