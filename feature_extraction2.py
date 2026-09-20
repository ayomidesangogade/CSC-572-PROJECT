"""
feature_extraction.py

Converts a raw URL into the 50 features used in the PhiUSIIL Phishing URL
Dataset, so a URL typed by a user can be fed into the model trained in
Phishing_URL_Detection_PhiUSIIL.ipynb.

IMPORTANT / LIMITATIONS
------------------------
Unlike the old UCI-dataset extractor, this one needs no WHOIS or DNS
lookups at all -- PhiUSIIL's feature set doesn't use domain age or DNS
records. It only needs to fetch the page itself (requests) and parse its
HTML (BeautifulSoup) and hostname (tldextract).

Three features can't be computed exactly the way the original PhiUSIIL
paper (Prasad & Chandra, 2024) does, because that requires their internal
reference database and undisclosed exact algorithm. Each is approximated
here using small lookup tables *derived directly from the training CSV
you supplied* (bundled as JSON files next to this script), not guessed:

  - TLDLegitimateProb: the paper's column correlates at 0.995 with a
    simple empirical P(TLD | legitimate) computed from the training data
    (verified during development) -- so this one is a faithful
    reconstruction, not a rough guess.
  - URLCharProb: approximated as the mean per-character frequency (from
    legitimate-class domains in the training data) of the characters in
    this URL's domain. The paper's exact character-probability model is
    not public.
  - URLSimilarityIndex: this needs a bigger caveat than the other two.
    In the training data, EVERY legitimate row has this value at exactly
    100 (zero variance), and every one of the 134,850 legitimate domains
    is unique -- i.e. the dataset authors appear to have assigned 100 to
    a URL once they already knew it was legitimate, rather than computing
    a genuine similarity score against some external reference. That
    makes this feature effectively an oracle: it encodes "is this a URL
    we already know is good," which cannot be honestly reproduced for a
    URL nobody has vetted yet. Since this is also the single strongest
    predictor in the trained model, expect the live app to under-score
    genuinely legitimate but less-common sites (anything not in the
    bundled reference list below) and lean toward "phishing" more than
    you'd want. This is a real limitation of the PhiUSIIL dataset for
    live deployment, not a bug in this script -- worth naming explicitly
    if you write about this in your thesis. The best available stand-in
    implemented here: an exact match against a bundled list of ~20,000
    known-legitimate domains (sampled from the training data) scores
    100; anything else falls back to fuzzy string matching, scaled to
    0-100.

All three are documented inline below at the point they're computed.
"""

import ipaddress
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

import requests
import tldextract
from bs4 import BeautifulSoup

try:
    from rapidfuzz import fuzz as _rf_fuzz  # optional, much faster; see requirements.txt
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

REQUEST_TIMEOUT = 6
HEADERS = {"User-Agent": "Mozilla/5.0 (phishing-detector-coursework-project)"}

_DATA_DIR = Path(__file__).parent / "reference_data"
_TLD_PROB = json.loads((_DATA_DIR / "tld_legit_prob.json").read_text())
_CHAR_PROB = json.loads((_DATA_DIR / "char_prob.json").read_text())
_REFERENCE_DOMAINS = json.loads((_DATA_DIR / "reference_domains.json").read_text())
_MIN_TLD_PROB = min(_TLD_PROB.values()) if _TLD_PROB else 1e-4
_MIN_CHAR_PROB = min(_CHAR_PROB.values()) if _CHAR_PROB else 1e-4

_REFERENCE_DOMAIN_SET = set(_REFERENCE_DOMAINS)  # O(1) exact-match fast path
_REFERENCE_BY_LEN = defaultdict(list)  # bounds the cost of the fuzzy fallback
for _d in _REFERENCE_DOMAINS:
    _REFERENCE_BY_LEN[len(_d)].append(_d)
_FUZZY_MAX_LEN_DIFF = 4

# Use only the bundled public-suffix-list snapshot (no network fetch needed),
# same approach as the original project's extractor.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

SOCIAL_DOMAINS = (
    "facebook.com", "twitter.com", "x.com", "instagram.com", "linkedin.com",
    "youtube.com", "tiktok.com", "pinterest.com", "reddit.com",
    "whatsapp.com", "telegram.org", "snapchat.com",
)
BANK_KEYWORDS = ("bank", "banking", "routing number", "iban", "swift code", "account number")
PAY_KEYWORDS = ("payment", "checkout", "credit card", "debit card", "paypal", "billing", "invoice")
CRYPTO_KEYWORDS = ("bitcoin", "ethereum", "crypto", "wallet address", "blockchain", "binance", "metamask")
POPUP_PATTERN = re.compile(r"window\.open\s*\(|alert\s*\(|confirm\s*\(|prompt\s*\(", re.IGNORECASE)
EMPTY_HREF = ("", "#", "javascript:void(0)", "javascript:void(0);")


def _registered_domain(hostname):
    """'ads.google.com' -> 'google.com', matching the original extractor's helper."""
    ext = _TLD_EXTRACTOR(hostname or "")
    return (ext.registered_domain or hostname or "").lower()


def _char_class(ch):
    if ch.isalpha():
        return "letter"
    if ch.isdigit():
        return "digit"
    return "other"


class PhishingFeatureExtractor:
    """Fetches a URL once and derives all 50 PhiUSIIL-schema features from it."""

    FEATURE_ORDER = [
        "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex",
        "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "TLDLength",
        "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
        "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
        "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
        "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
        "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
        "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
        "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
        "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
        "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
        "NoOfEmptyRef", "NoOfExternalRef",
    ]

    def __init__(self, url):
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "https://" + url
        self.url = url
        parsed = urlparse(self.url)
        self.hostname = parsed.hostname or ""
        self.registered_domain = _registered_domain(self.hostname)
        ext = _TLD_EXTRACTOR(self.hostname)
        self.tld = ext.suffix or ""

        self.html = ""
        self.soup = BeautifulSoup("", "html.parser")
        self.response = None
        self.redirect_history = []
        self._fetch_page()

    # ---------- networking ----------
    def _do_request(self, url):
        self.response = requests.get(
            url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )
        self.html = self.response.text
        self.redirect_history = self.response.history
        self.soup = BeautifulSoup(self.html, "html.parser")

    def _fetch_page(self):
        try:
            self._do_request(self.url)
            return
        except requests.RequestException:
            pass
        if self.url.startswith("https://"):
            fallback_url = self.url.replace("https://", "http://", 1)
            try:
                self._do_request(fallback_url)
                self.url = fallback_url
                return
            except requests.RequestException:
                pass
        # Both attempts failed: leave self.html/self.soup as empty defaults,
        # every feature below is written to degrade gracefully in that case.

    # ---------- lexical features (no network needed) ----------
    def URLLength(self):
        return len(self.url)

    def DomainLength(self):
        return len(self.hostname)

    def IsDomainIP(self):
        try:
            ipaddress.ip_address(self.hostname)
            return 1
        except ValueError:
            return 0

    def TLDLength(self):
        return len(self.tld)

    def NoOfSubDomain(self):
        ext = _TLD_EXTRACTOR(self.hostname)
        if not ext.subdomain:
            return 0
        return ext.subdomain.count(".") + 1

    def HasObfuscation(self):
        return 1 if re.search(r"%[0-9A-Fa-f]{2}", self.url) else 0

    def NoOfObfuscatedChar(self):
        return len(re.findall(r"%[0-9A-Fa-f]{2}", self.url))

    def ObfuscationRatio(self):
        n = len(self.url)
        return (self.NoOfObfuscatedChar() * 3 / n) if n else 0.0

    def NoOfLettersInURL(self):
        return sum(c.isalpha() for c in self.url)

    def LetterRatioInURL(self):
        n = len(self.url)
        return self.NoOfLettersInURL() / n if n else 0.0

    def NoOfDegitsInURL(self):
        return sum(c.isdigit() for c in self.url)

    def DegitRatioInURL(self):
        n = len(self.url)
        return self.NoOfDegitsInURL() / n if n else 0.0

    def NoOfEqualsInURL(self):
        return self.url.count("=")

    def NoOfQMarkInURL(self):
        return self.url.count("?")

    def NoOfAmpersandInURL(self):
        return self.url.count("&")

    def NoOfOtherSpecialCharsInURL(self):
        specials = sum(not c.isalnum() for c in self.url)
        # subtract the ones already counted by their own dedicated features
        return specials - self.NoOfEqualsInURL() - self.NoOfQMarkInURL() - self.NoOfAmpersandInURL()

    def SpacialCharRatioInURL(self):
        n = len(self.url)
        if not n:
            return 0.0
        specials = sum(not c.isalnum() for c in self.url)
        return specials / n

    def IsHTTPS(self):
        return 1 if self.url.lower().startswith("https://") else 0

    def CharContinuationRate(self):
        s = self.url
        if len(s) < 2:
            return 0.0
        same = sum(_char_class(s[i]) == _char_class(s[i - 1]) for i in range(1, len(s)))
        return same / (len(s) - 1)

    # ---------- empirically-derived features (bundled lookup tables; see module docstring) ----------
    def TLDLegitimateProb(self):
        return _TLD_PROB.get(self.tld, _MIN_TLD_PROB)

    def URLCharProb(self):
        domain = self.hostname.lower()
        if not domain:
            return _MIN_CHAR_PROB
        probs = [_CHAR_PROB.get(c, _MIN_CHAR_PROB) for c in domain]
        return sum(probs) / len(probs)

    def URLSimilarityIndex(self):
        # See the module docstring: this is a best-effort stand-in, not a
        # reproduction of the paper's method.
        host = self.hostname.lower()
        if not host:
            return 0
        if host in _REFERENCE_DOMAIN_SET:
            return 100  # exact match -> treat like the training data's legitimate rows

        n = len(host)
        candidates = []
        for length in range(max(1, n - _FUZZY_MAX_LEN_DIFF), n + _FUZZY_MAX_LEN_DIFF + 1):
            candidates.extend(_REFERENCE_BY_LEN.get(length, ()))
        if not candidates:
            candidates = _REFERENCE_DOMAINS  # very short/long host: widen the search

        if _HAS_RAPIDFUZZ:
            best = max((_rf_fuzz.ratio(host, ref) for ref in candidates), default=0.0)
            return round(best)
        best = 0.0
        for ref in candidates:
            ratio = SequenceMatcher(None, host, ref).ratio()
            if ratio > best:
                best = ratio
        return round(best * 100)

    # ---------- page-content features (require a successful fetch) ----------
    def LineOfCode(self):
        return self.html.count("\n") + 1 if self.html else 0

    def LargestLineLength(self):
        lines = self.html.splitlines()
        return max((len(l) for l in lines), default=0)

    def _title_text(self):
        return self.soup.title.get_text(strip=True) if self.soup.title else ""

    def HasTitle(self):
        return 1 if self._title_text() else 0

    def DomainTitleMatchScore(self):
        title = self._title_text().lower()
        if not title:
            return 0
        return round(SequenceMatcher(None, self.hostname.lower(), title).ratio() * 100)

    def URLTitleMatchScore(self):
        title = self._title_text().lower()
        if not title:
            return 0
        return round(SequenceMatcher(None, self.url.lower(), title).ratio() * 100)

    def HasFavicon(self):
        return 1 if self.soup.find("link", rel=lambda v: v and "icon" in v.lower()) else 0

    def Robots(self):
        has_meta = self.soup.find("meta", attrs={"name": lambda v: v and v.lower() == "robots"})
        return 1 if has_meta else 0

    def IsResponsive(self):
        viewport = self.soup.find("meta", attrs={"name": lambda v: v and v.lower() == "viewport"})
        return 1 if viewport and "width=device-width" in (viewport.get("content") or "").lower() else 0

    def NoOfURLRedirect(self):
        return len(self.redirect_history)

    def NoOfSelfRedirect(self):
        count = 0
        for resp in self.redirect_history:
            loc = resp.headers.get("Location", "")
            if loc and self.registered_domain and self.registered_domain in loc.lower():
                count += 1
        return count

    def HasDescription(self):
        desc = self.soup.find("meta", attrs={"name": lambda v: v and v.lower() == "description"})
        return 1 if desc and (desc.get("content") or "").strip() else 0

    def NoOfPopup(self):
        return len(POPUP_PATTERN.findall(self.html))

    def NoOfiFrame(self):
        return len(self.soup.find_all("iframe"))

    def _forms(self):
        return self.soup.find_all("form")

    def HasExternalFormSubmit(self):
        for form in self._forms():
            action = (form.get("action") or "").strip()
            if action.startswith("http") and self.registered_domain not in action.lower():
                return 1
        return 0

    def HasSocialNet(self):
        for a in self.soup.find_all("a"):
            href = (a.get("href") or "").lower()
            if any(s in href for s in SOCIAL_DOMAINS):
                return 1
        return 0

    def HasSubmitButton(self):
        if self.soup.find("button", attrs={"type": lambda v: v and v.lower() == "submit"}):
            return 1
        if self.soup.find("input", attrs={"type": lambda v: v and v.lower() == "submit"}):
            return 1
        return 0

    def HasHiddenFields(self):
        return 1 if self.soup.find("input", attrs={"type": lambda v: v and v.lower() == "hidden"}) else 0

    def HasPasswordField(self):
        return 1 if self.soup.find("input", attrs={"type": lambda v: v and v.lower() == "password"}) else 0

    def _page_text(self):
        return self.soup.get_text(separator=" ").lower()

    def Bank(self):
        text = self._page_text()
        return 1 if any(k in text for k in BANK_KEYWORDS) else 0

    def Pay(self):
        text = self._page_text()
        return 1 if any(k in text for k in PAY_KEYWORDS) else 0

    def Crypto(self):
        text = self._page_text()
        return 1 if any(k in text for k in CRYPTO_KEYWORDS) else 0

    def HasCopyrightInfo(self):
        # Use the BeautifulSoup-decoded text, not the raw HTML string, so
        # both the literal "©" glyph and the "&copy;"/"&#169;" HTML entity
        # forms are caught (BeautifulSoup decodes entities automatically).
        text = self._page_text()
        return 1 if ("©" in text or "copyright" in text) else 0

    def NoOfImage(self):
        return len(self.soup.find_all("img"))

    def NoOfCSS(self):
        links = self.soup.find_all("link", rel=lambda v: v and "stylesheet" in v.lower())
        styles = self.soup.find_all("style")
        return len(links) + len(styles)

    def NoOfJS(self):
        return len(self.soup.find_all("script"))

    def _classify_refs(self):
        self_ref = empty_ref = external_ref = 0
        for a in self.soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if href.lower() in EMPTY_HREF or href.startswith("#") or href.lower().startswith("javascript:"):
                empty_ref += 1
            elif href.startswith("http") and self.registered_domain not in href.lower():
                external_ref += 1
            else:
                self_ref += 1
        return self_ref, empty_ref, external_ref

    def NoOfSelfRef(self):
        return self._classify_refs()[0]

    def NoOfEmptyRef(self):
        return self._classify_refs()[1]

    def NoOfExternalRef(self):
        return self._classify_refs()[2]

    # ---------- public API ----------
    def extract(self):
        values = {}
        for name in self.FEATURE_ORDER:
            method = getattr(self, name)
            try:
                values[name] = method()
            except Exception:
                values[name] = 0
        return values

    def extract_vector(self, columns):
        values = self.extract()
        return [values[col] for col in columns]

    def fetch_succeeded(self):
        return bool(self.html)


def extract_features(url):
    """Convenience function: returns (feature_dict, extractor_instance)."""
    extractor = PhishingFeatureExtractor(url)
    return extractor.extract(), extractor
