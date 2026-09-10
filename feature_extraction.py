"""
feature_extraction.py

Converts a raw URL into the same 30 features used in the UCI Phishing
Websites dataset, so that a URL typed by a user can be fed into the
trained model in real time.

IMPORTANT / LIMITATIONS
------------------------
The original UCI dataset included a small number of features that relied
on third-party services which have since been discontinued or paywalled:
    - web_traffic      (originally Alexa rank; Alexa shut down in 2022)
    - Page_Rank         (originally Google's public PageRank API; discontinued)
    - Google_Index      (originally scraped Google search; against Google's ToS)
    - Links_pointing_to_page (originally a backlink/SEO API)
    - Statistical_report (originally cross-referenced PhishTank/StopBadware stats DBs)

For these five features we fall back to a fixed neutral/default value
(documented inline below) rather than making a live call. Every other
feature is computed live from the URL, DNS, WHOIS, SSL, and page HTML.

All feature functions return values on the same {-1, 0, 1} (or {0, 1})
scale used in the original dataset so the vector can be passed straight
into the trained classifier.
"""

import re
import socket
import ssl
import ipaddress
from datetime import datetime
from urllib.parse import urlparse

import requests
import whois
import dns.resolver
import tldextract
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 5
HEADERS = {"User-Agent": "Mozilla/5.0 (phishing-detector-coursework-project)"}

# Use only the bundled public-suffix-list snapshot (no network fetch needed)
# so this works reliably offline and doesn't add a dependency on yet another
# third-party service just to parse domain names.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


def _registered_domain(hostname):
    """
    Returns the registrable base domain for a hostname, e.g. both
    "www.google.com" and "ads.google.com" -> "google.com". Comparing
    against this, rather than the exact hostname the user typed, avoids
    flagging a site's own sibling subdomains (ads.google.com,
    policies.google.com, support.google.com, etc.) as "external" links.
    """
    ext = _TLD_EXTRACTOR(hostname or "")
    return (ext.registered_domain or hostname or "").lower()

SHORTENING_SERVICES = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|"
    r"is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|"
    r"su\.pr|twurl\.nl|snipurl\.com|short\.to|budurl\.com|ping\.fm|post\.ly|"
    r"just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|"
    r"kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|buff\.ly|adf\.ly|"
    r"lnkd\.in|db\.tt|qr\.ae|cur\.lv|tinyarrows\.com|v\.gd"
)


class PhishingFeatureExtractor:
    """Fetches a URL once and derives all 30 dataset features from it."""

    FEATURE_ORDER = [
        "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
        "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
        "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
        "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
        "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
        "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
        "Statistical_report",
    ]

    def __init__(self, url):
        if not re.match(r"^https?://", url, re.IGNORECASE):
            # Most legitimate modern sites serve HTTPS by default. Try HTTPS
            # first and only fall back to HTTP if that scheme is genuinely
            # unreachable, rather than assuming HTTP just because the user
            # omitted a scheme when typing the URL (e.g. "google.com").
            url = "https://" + url
        self.url = url
        parsed = urlparse(self.url)
        self.scheme = parsed.scheme
        self.hostname = parsed.hostname or ""
        self.registered_domain = _registered_domain(self.hostname)
        self.port_in_url = parsed.port
        self.path = parsed.path or ""

        self.html = ""
        self.soup = None
        self.response = None
        self.redirect_count = 0
        self._fetch_page()

        self.whois_data = self._get_whois()
        self.notes = []  # human-readable notes for the demo app to display

    # ---------- networking helpers ----------
    def _do_request(self, url):
        self.response = requests.get(
            url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )
        self.html = self.response.text
        self.redirect_count = len(self.response.history)
        self.soup = BeautifulSoup(self.html, "html.parser")
        # Reflect the final, post-redirect scheme (e.g. an http -> https
        # upgrade performed by the server) rather than the scheme the
        # request started with.
        self.scheme = urlparse(self.response.url).scheme

    def _fetch_page(self):
        try:
            self._do_request(self.url)
            return
        except requests.RequestException:
            pass

        if self.scheme == "https":
            # The HTTPS attempt failed outright (connection refused, no
            # listener, etc.) - fall back to HTTP before giving up, since
            # a small number of legitimate sites still only serve plain
            # HTTP. This does NOT affect the SSLfinal_State feature below,
            # which always tests the real certificate independently.
            fallback_url = self.url.replace("https://", "http://", 1)
            try:
                self._do_request(fallback_url)
                self.url = fallback_url
                return
            except requests.RequestException:
                pass

        self.response = None
        self.html = ""
        self.soup = BeautifulSoup("", "html.parser")

    def _get_whois(self):
        try:
            return whois.whois(self.hostname)
        except Exception:
            return None

    @staticmethod
    def _first(value):
        """WHOIS sometimes returns a list of dates; take the earliest/first."""
        if isinstance(value, list):
            return value[0] if value else None
        return value

    # ---------- lexical features ----------
    def having_IP_Address(self):
        # Empirically, in the training data, an IP-address host is more
        # associated with phishing, which is coded as the LOWER value
        # for this feature (an earlier version of this module had this
        # backward).
        try:
            ipaddress.ip_address(self.hostname)
            return -1
        except ValueError:
            return 1

    def URL_Length(self):
        length = len(self.url)
        if length < 54:
            return 1
        elif length <= 75:
            return 0
        return -1

    def Shortining_Service(self):
        return 1 if SHORTENING_SERVICES.search(self.url) else -1

    def having_At_Symbol(self):
        return -1 if "@" in self.url else 1

    def double_slash_redirecting(self):
        return 1 if self.url.rfind("//") > 7 else -1

    def Prefix_Suffix(self):
        return 1 if "-" in self.hostname else -1

    def having_Sub_Domain(self):
        # Empirically, a simple domain with few subdomains is associated
        # with the HIGHER value for this feature (an earlier version of
        # this module had this backward).
        host = self.hostname
        if host.startswith("www."):
            host = host[4:]
        dot_count = host.count(".")
        if dot_count <= 1:
            return 1
        elif dot_count == 2:
            return 0
        return -1

    def port(self):
        standard_ports = {80, 443, None}
        return 1 if self.port_in_url not in standard_ports else -1

    def HTTPS_token(self):
        return 1 if "https" in self.hostname.lower() else -1

    # ---------- SSL / domain features ----------
    def SSLfinal_State(self):
        """
        Always attempts a real TLS handshake on port 443, independently of
        which scheme the content-fetch above ended up using. Empirically,
        a valid trusted certificate is associated with the HIGHER value
        for this feature (an earlier version of this module had this
        backward, returning -1 for a valid cert instead of 1).
        """
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((self.hostname, 443), timeout=REQUEST_TIMEOUT) as sock:
                with ctx.wrap_socket(sock, server_hostname=self.hostname):
                    return 1  # connected and certificate is trusted
        except ssl.SSLCertVerificationError:
            return 0  # SSL is present but the certificate is invalid/self-signed
        except Exception:
            return -1  # no SSL/TLS available on port 443 at all

    def Domain_registeration_length(self):
        if not self.whois_data:
            return 1
        try:
            exp = self._first(self.whois_data.expiration_date)
            created = self._first(self.whois_data.creation_date)
            if not exp or not created:
                return 1
            age_days = (exp - created).days
            return -1 if age_days >= 365 else 1
        except Exception:
            return 1

    def age_of_domain(self):
        if not self.whois_data:
            return -1
        try:
            created = self._first(self.whois_data.creation_date)
            if not created:
                return -1
            months = (datetime.now() - created).days / 30
            return 1 if months >= 6 else -1
        except Exception:
            return -1

    def DNSRecord(self):
        try:
            dns.resolver.resolve(self.hostname, "A")
            return 1
        except Exception:
            return -1

    def Abnormal_URL(self):
        if not self.whois_data:
            return 1
        try:
            registered = str(self.whois_data.domain_name or "").lower()
            return -1 if self.hostname.lower() in registered or registered in self.hostname.lower() else 1
        except Exception:
            return 1

    def Redirect(self):
        return 1 if self.redirect_count >= 2 else 0

    # ---------- page-content features ----------
    def Favicon(self):
        icon = self.soup.find("link", rel=lambda v: v and "icon" in v.lower())
        if not icon or not icon.get("href"):
            return -1
        href = icon["href"]
        return 1 if href.startswith("http") and self.registered_domain not in href.lower() else -1

    def _external_ratio(self, tags_and_attr):
        total, external = 0, 0
        for tag, attr in tags_and_attr:
            for el in self.soup.find_all(tag):
                val = el.get(attr)
                if not val:
                    continue
                total += 1
                if val.startswith("http") and self.registered_domain not in val.lower():
                    external += 1
        if total == 0:
            return -1
        ratio = external / total
        return ratio

    def Request_URL(self):
        # Empirically, a HIGH proportion of same-site media/resource links
        # is associated with the HIGHER value for this feature (an earlier
        # version of this module had this backward).
        ratio = self._external_ratio([("img", "src"), ("audio", "src"), ("embed", "src"), ("iframe", "src")])
        if ratio == -1:
            return 1
        if ratio > 0.61:
            return -1
        elif ratio >= 0.22:
            return 0
        return 1

    def URL_of_Anchor(self):
        # Empirically, a LOW proportion of empty/external anchor links is
        # associated with the HIGHER value for this feature (an earlier
        # version of this module had this backward). A page with zero
        # anchor tags at all is unusual for a normal website and is left
        # as phishing-leaning by default, rather than defaulting to
        # legitimate just because there was nothing to measure.
        anchors = self.soup.find_all("a")
        if not anchors:
            return -1
        suspicious = 0
        for a in anchors:
            href = a.get("href", "")
            if not href or href.startswith("#") or href.lower().startswith("javascript:"):
                suspicious += 1
            elif href.startswith("http") and self.registered_domain not in href.lower():
                suspicious += 1
        ratio = suspicious / len(anchors)
        if ratio > 0.67:
            return -1
        elif ratio >= 0.31:
            return 0
        return 1

    def Links_in_tags(self):
        ratio = self._external_ratio([("meta", "content"), ("script", "src"), ("link", "href")])
        if ratio == -1:
            return 1
        if ratio > 0.81:
            return -1
        elif ratio >= 0.17:
            return 0
        return 1

    def SFH(self):
        # Empirically, a form that submits to the same site is associated
        # with the HIGHER value for this feature (an earlier version of
        # this module had this backward).
        form = self.soup.find("form")
        if not form:
            return 1
        action = form.get("action", "")
        if action.strip() in ("", "about:blank"):
            return -1
        if action.startswith("http") and self.registered_domain not in action.lower():
            return 0
        return 1

    def Submitting_to_email(self):
        return 1 if "mailto:" in self.html.lower() else -1

    def on_mouseover(self):
        return 1 if re.search(r"onmouseover\s*=.*window\.status", self.html, re.IGNORECASE) else -1

    def RightClick(self):
        return 1 if re.search(r"event\.button\s*==\s*2|contextmenu", self.html, re.IGNORECASE) else -1

    def popUpWidnow(self):
        return 1 if re.search(r"alert\s*\(|prompt\s*\(", self.html, re.IGNORECASE) else -1

    def Iframe(self):
        return 1 if "<iframe" in self.html.lower() else -1

    # ---------- features approximated due to discontinued third-party APIs ----------
    def web_traffic(self):
        # Originally sourced from Alexa rank (service discontinued in 2022).
        return 0

    def Page_Rank(self):
        # Originally Google's public PageRank API (discontinued).
        return -1

    def Google_Index(self):
        # Originally checked whether Google had indexed the page.
        return 1

    def Links_pointing_to_page(self):
        # Originally a backlink count from an SEO API.
        return 0

    def Statistical_report(self):
        # Originally cross-referenced PhishTank / StopBadware statistical reports.
        return -1

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


def extract_features(url):
    """Convenience function: returns (feature_dict, extractor_instance)."""
    extractor = PhishingFeatureExtractor(url)
    return extractor.extract(), extractor
