"""
common_lexical_features.py

Computes the 9 UCI-style features that can be derived purely from a URL
string (no DNS/WHOIS/SSL/HTML fetch required), so they can be applied to
BOTH the 2014 UCI dataset (already encoded this way) and the 2024 PhiUSIIL
dataset's raw "URL" column, giving a genuinely common, identically-defined
feature space for the cross-dataset generalisation experiment.

Every function body below is copied verbatim from the logic in
feature_extraction.py (same thresholds, same regex, same sign convention)
so this is not a re-derivation -- it's the same feature definitions the
model was trained on, minus the parts that require a live network call.

Excluded from the common feature set (and why):
    - SSLfinal_State, URL_of_Anchor, Request_URL, Links_in_tags, SFH,
      Favicon: require live page content / TLS handshake, not available
      for 235k historical PhiUSIIL URLs, and PhiUSIIL's own precomputed
      analogs (IsHTTPS, HasExternalFormSubmit, HasFavicon, etc.) use a
      different definition than the UCI feature, so mapping them would
      not be a like-for-like comparison.
    - Domain_registeration_length, age_of_domain, DNSRecord, Abnormal_URL:
      require WHOIS/DNS lookups against potentially dead/historical hosts.
    - web_traffic, Page_Rank, Google_Index, Links_pointing_to_page,
      Statistical_report: already approximated with fixed defaults even
      in the original real-time app (discontinued third-party services).
"""

import re
import ipaddress
from urllib.parse import urlparse

SHORTENING_SERVICES = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|"
    r"is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|"
    r"su\.pr|twurl\.nl|snipurl\.com|short\.to|budurl\.com|ping\.fm|post\.ly|"
    r"just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|"
    r"kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|buff\.ly|adf\.ly|"
    r"lnkd\.in|db\.tt|qr\.ae|cur\.lv|tinyarrows\.com|v\.gd"
)

#
# VALIDATION NOTE (added after empirically checking each feature against
# UCI's own conditional phishing rate, i.e. df.groupby(col)['Result'].mean()):
#
#   - Prefix_Suffix: UCI value=1 is 100% legitimate (n=1465), value=-1 is
#     ~51% phishing. The function below originally had this backwards
#     (matching the same kind of sign bug already documented in Chapter 5,
#     Limitation #6, for SSLfinal_State/URL_of_Anchor). FIXED below.
#   - port: UCI's majority value (86%) must correspond to "standard port",
#     since the overwhelming majority of real URLs never specify a port.
#     The function below originally had this backwards. FIXED below.
#   - Shortining_Service, double_slash_redirecting, HTTPS_token: UCI's
#     majority class for all three (84-87%) is not plausible as a literal
#     match against a short regex/substring check (e.g. 87% of all URLs
#     cannot really contain a known link-shortener domain). This means
#     UCI's actual original feature definition is NOT what a direct
#     re-implementation computes, and there is no reliable way to recover
#     it. Per the plan's own guidance ("exclude features that cannot be
#     mapped reliably; do not invent values"), these 3 are EXCLUDED from
#     the common set rather than guessed at.
#
COMMON_FEATURES = [
    "having_IP_Address", "URL_Length", "having_At_Symbol",
    "Prefix_Suffix", "having_Sub_Domain", "port",
]


def compute_common_features(url):
    """Given a raw URL string, return a dict of the 9 common features,
    on the same {-1, 0, 1} scale as the UCI dataset."""
    if not re.match(r"^https?://", str(url), re.IGNORECASE):
        url = "https://" + str(url)
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    port_in_url = parsed.port

    out = {}

    # having_IP_Address
    try:
        ipaddress.ip_address(hostname)
        out["having_IP_Address"] = -1
    except ValueError:
        out["having_IP_Address"] = 1

    # URL_Length
    length = len(url)
    if length < 54:
        out["URL_Length"] = 1
    elif length <= 75:
        out["URL_Length"] = 0
    else:
        out["URL_Length"] = -1

    # Shortining_Service
    out["Shortining_Service"] = 1 if SHORTENING_SERVICES.search(url) else -1

    # having_At_Symbol
    out["having_At_Symbol"] = -1 if "@" in url else 1

    # double_slash_redirecting
    out["double_slash_redirecting"] = 1 if url.rfind("//") > 7 else -1

    # Prefix_Suffix (sign fixed: UCI's value=1 is ~100% legitimate, i.e.
    # 1 = no hyphen / safe, -1 = hyphen present / risky)
    out["Prefix_Suffix"] = -1 if "-" in hostname else 1

    # having_Sub_Domain
    host = hostname
    if host.startswith("www."):
        host = host[4:]
    dot_count = host.count(".")
    if dot_count <= 1:
        out["having_Sub_Domain"] = 1
    elif dot_count == 2:
        out["having_Sub_Domain"] = 0
    else:
        out["having_Sub_Domain"] = -1

    # port (sign fixed: UCI's majority value=1 must be "standard port",
    # since the vast majority of real URLs never specify a port)
    standard_ports = {80, 443, None}
    out["port"] = -1 if port_in_url not in standard_ports else 1

    # HTTPS_token
    out["HTTPS_token"] = 1 if "https" in hostname.lower() else -1

    return out
