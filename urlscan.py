"""URL analysis: safety heuristics + review extraction from a product/review page."""
import json
import os
import re
from urllib.parse import urlparse, parse_qs

import pandas as pd

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    requests = None
    BeautifulSoup = None

PLATFORMS = {
    "amazon": "Amazon", "flipkart": "Flipkart", "myntra": "Myntra", "meesho": "Meesho",
    "snapdeal": "Snapdeal", "ebay": "eBay", "walmart": "Walmart", "tripadvisor": "TripAdvisor",
    "yelp": "Yelp", "google": "Google", "booking": "Booking.com", "zomato": "Zomato",
    "nykaa": "Nykaa", "ajio": "Ajio", "bestbuy": "Best Buy", "aliexpress": "AliExpress",
}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "ow.ly", "rb.gy", "shorturl.at"}
RISKY_TLDS = {"xyz", "top", "click", "shop", "online", "site", "icu", "tk", "ml", "ga", "cf", "gq", "work", "live", "buzz"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept-Language": "en-US,en;q=0.9"}


def normalise(url):
    url = (url or "").strip()
    if not url:
        raise ValueError("Please enter a URL.")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    p = urlparse(url)
    if not p.netloc or "." not in p.netloc:
        raise ValueError("That does not look like a valid URL.")
    return url


def url_risk(url):
    """Heuristic safety / authenticity check of the URL itself. Returns dict."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    parts = host.split(".")
    tld = parts[-1] if parts else ""
    base = ".".join(parts[-2:]) if len(parts) >= 2 else host
    platform = next((v for k, v in PLATFORMS.items() if k in host), "Unknown")
    checks, risk = [], 0

    def add(ok, good, bad, pts):
        nonlocal risk
        checks.append({"ok": ok, "text": good if ok else bad})
        if not ok:
            risk += pts

    add(p.scheme == "https", "Uses HTTPS encryption", "Not using HTTPS", 20)
    add(not re.fullmatch(r"[\d.]+", host), "Domain is a name, not a raw IP address", "Raw IP address used as host", 25)
    add(base not in SHORTENERS, "Not a link-shortener", "Link-shortener hides the real destination", 20)
    add(tld not in RISKY_TLDS, f"Common TLD (.{tld})", f"Higher-abuse TLD (.{tld})", 10)
    add("@" not in url and "xn--" not in host, "No '@' or punycode tricks", "Contains '@' or punycode (possible spoofing)", 25)
    add(len(parts) <= 4, "Normal sub-domain depth", "Unusually many sub-domains", 10)
    brand_spoof = any(k in host and not (host == f"{k}.com" or host.endswith(f".{k}.com") or host.endswith(f".{k}.in")
                      or host.endswith(f".{k}.co.uk") or host.endswith(f".{k}.org")) for k in PLATFORMS)
    add(not brand_spoof, "Domain matches a known platform pattern" if platform != "Unknown" else "No brand-impersonation pattern",
        "Looks like a look-alike of a well-known brand", 30)
    add(len(url) < 200, "URL length is normal", "Very long URL", 5)
    risk = min(risk, 100)
    return {"host": host, "platform": platform, "risk": risk,
            "level": "High" if risk >= 40 else ("Medium" if risk >= 20 else "Low"),
            "checks": checks, "product_id": product_id(url)}


def product_id(url):
    m = re.search(r"/(?:dp|gp/product|product-reviews)/([A-Z0-9]{10})", url)
    if m:
        return "ASIN " + m.group(1)
    q = parse_qs(urlparse(url).query)
    if "pid" in q:
        return "PID " + q["pid"][0]
    return None


BLOCK_HINTS = ("captcha", "robot check", "enter the characters you see", "automated access", "unusual traffic",
               "api-services-support@amazon.com", "access denied", "are you a human")


def fetch_html(url, timeout=15):
    """Download a page. If SCRAPERAPI_KEY is set, the request goes through that proxy service
    (needed on cloud hosts, whose IP addresses most marketplaces block)."""
    if requests is None:
        raise ValueError("The 'requests' and 'beautifulsoup4' packages are required for URL analysis.")
    key = os.environ.get("SCRAPERAPI_KEY")
    try:
        if key:
            r = requests.get("https://api.scraperapi.com/", params={"api_key": key, "url": url},
                             timeout=70)
        else:
            r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not reach the page ({type(e).__name__}). Check the link/internet connection.")
    head = r.text.lower()[:30000]
    blocked_msg = ("The site blocked this server with a CAPTCHA / bot check (common for cloud-hosted apps - "
                   "marketplaces trust home internet connections but block data-centre addresses). "
                   "Use the 'Upload saved page' option below, or paste the reviews into Bulk Analysis.")
    if r.status_code in (403, 429, 503):
        raise ValueError(blocked_msg)
    if r.status_code >= 400:
        raise ValueError(f"The page returned HTTP {r.status_code}.")
    if any(h in head for h in BLOCK_HINTS) and "data-hook=\"review\"" not in r.text and "application/ld+json" not in r.text:
        raise ValueError(blocked_msg)
    return r.text


def _num(x):
    m = re.search(r"\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else None


def extract_reviews(html):
    """Return DataFrame(review, rating, date, verified, title) from page HTML."""
    soup = BeautifulSoup(html, "lxml")
    rows = []

    # 1) JSON-LD (schema.org Review) - works on many sites
    def walk(o):
        if isinstance(o, dict):
            if str(o.get("@type", "")).lower() == "review":
                body = o.get("reviewBody") or o.get("description") or ""
                rr = o.get("reviewRating") or {}
                rows.append({"review": str(body), "title": str(o.get("name", "")),
                             "rating": _num(rr.get("ratingValue")) if isinstance(rr, dict) else None,
                             "date": o.get("datePublished"), "verified": None})
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            walk(json.loads(tag.string or ""))
        except (ValueError, TypeError):
            pass

    # 2) Amazon markup
    if not rows:
        for div in soup.select('[data-hook="review"]'):
            body = div.select_one('[data-hook="review-body"]')
            if not body:
                continue
            rt = div.select_one('[data-hook="review-star-rating"], [data-hook="cmps-review-star-rating"]')
            ttl = div.select_one('[data-hook="review-title"]')
            dt = div.select_one('[data-hook="review-date"]')
            rows.append({"review": body.get_text(" ", strip=True), "rating": _num(rt.get_text()) if rt else None,
                         "title": ttl.get_text(" ", strip=True) if ttl else "",
                         "date": dt.get_text(" ", strip=True).split(" on ")[-1] if dt else None,
                         "verified": bool(div.select_one('[data-hook="avp-badge"]'))})

    # 3) schema.org microdata
    if not rows:
        for el in soup.select('[itemprop="review"], [itemtype*="schema.org/Review"]'):
            body = el.select_one('[itemprop="reviewBody"], [itemprop="description"]')
            if body:
                rt = el.select_one('[itemprop="ratingValue"]')
                rows.append({"review": body.get_text(" ", strip=True),
                             "rating": _num(rt.get("content") or rt.get_text()) if rt else None,
                             "title": "", "date": None, "verified": None})

    # 4) Generic: Flipkart / others - blocks whose class mentions review/comment
    if not rows:
        seen = set()
        for el in soup.select('[class*="review"], [class*="Review"], [class*="comment"], [class*="ZmyHeo"], [class*="t-ZTKy"]'):
            if el.find(attrs={"class": re.compile("review|comment", re.I)}) and len(list(el.children)) > 6:
                continue  # container, not a single review
            t = el.get_text(" ", strip=True)
            t = re.sub(r"\s*READ MORE\s*$", "", t)
            if 25 <= len(t) <= 3000 and t not in seen:
                seen.add(t)
                rows.append({"review": t, "rating": None, "title": "", "date": None, "verified": None})

    df = pd.DataFrame(rows, columns=["review", "title", "rating", "date", "verified"])
    df = df[df["review"].str.strip().str.len() > 10].drop_duplicates("review").reset_index(drop=True)
    return df


def scan_url(url):
    url = normalise(url)
    info = url_risk(url)
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")
    info["page_title"] = (soup.title.get_text(strip=True) if soup.title else "")[:140]
    df = extract_reviews(html)
    if df.empty:
        raise ValueError("The page loaded but no reviews could be found in its HTML (most marketplaces load "
                         "reviews with JavaScript). Copy the reviews and use Paste / Bulk Upload - the analysis is identical.")
    return url, info, df


def scan_html(html, url=""):
    """Analyse a page the user saved/pasted themselves (no network request needed)."""
    url = normalise(url) if url.strip() else "https://saved-page.local/"
    info = url_risk(url) if url != "https://saved-page.local/" else {"host": "saved page", "platform": "Saved page",
        "risk": 0, "level": "Low", "checks": [], "product_id": None}
    soup = BeautifulSoup(html, "lxml")
    info["page_title"] = (soup.title.get_text(strip=True) if soup.title else "Saved page")[:140]
    if url != "https://saved-page.local/":
        pass
    df = extract_reviews(html)
    if df.empty:
        raise ValueError("No reviews were found in that page. Open the page with the reviews visible "
                         "(scroll so they load), save it as 'Webpage, HTML only' and upload it again - "
                         "or copy the reviews into Bulk Analysis.")
    return url, info, df


def scan_sample():
    """Built-in demo page (works offline and on any host)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "sample_product_page.html")
    with open(p, encoding="utf-8") as f:
        html = f.read()
    return scan_html(html, "https://www.fitpulse-shop.com/products/x1-smartwatch")
