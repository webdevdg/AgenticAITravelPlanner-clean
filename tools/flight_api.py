# tools/flight_api.py
import os
import time
import requests
from typing import List, Dict, Any, Optional, Iterable

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ---------- Amadeus token cache ----------
_TOKEN: Dict[str, Any] = {"access_token": None, "expires_at": 0.0}

def get_amadeus_access_token() -> str:
    """
    Get (and cache) an Amadeus OAuth token. Reuses it until expiry.
    Requires AMADEUS_CLIENT_ID / AMADEUS_CLIENT_SECRET in the environment.
    """
    now = time.time()
    if _TOKEN["access_token"] and now < _TOKEN["expires_at"]:
        return _TOKEN["access_token"]

    cid = os.getenv("AMADEUS_CLIENT_ID")
    cs  = os.getenv("AMADEUS_CLIENT_SECRET")
    if not cid or not cs:
        raise ValueError("AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET must be set in environment variables.")
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {"grant_type": "client_credentials", "client_id": cid, "client_secret": cs}
    resp = requests.post(url, data=data, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    _TOKEN["access_token"] = payload["access_token"]
    _TOKEN["expires_at"]   = now + float(payload.get("expires_in", 1800)) - 60  # refresh a minute early
    return _TOKEN["access_token"]

# ---------- Helpers ----------
def resolve_loc_code(term: str, token: str) -> Optional[str]:
    """
    Accepts a city/airport NAME (e.g., 'New York', 'JFK') or CODE ('NYC', 'JFK').
    Returns a 3-letter IATA city/airport code or None if not found.
    """
    if not term:
        return None
    t = term.strip().upper()
    if len(t) == 3 and t.isalpha():
        return t  # already code
    url = "https://test.api.amadeus.com/v1/reference-data/locations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"subType": "CITY,AIRPORT", "keyword": term}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return None
    return data[0].get("iataCode")

def _num(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _flatten_segments(itinerary: Dict[str, Any], carriers: Dict[str, str]) -> List[Dict[str, Any]]:
    segs_out: List[Dict[str, Any]] = []
    for seg in itinerary.get("segments", []) or []:
        dep = seg.get("departure", {}) or {}
        arr = seg.get("arrival", {}) or {}
        code = seg.get("carrierCode")
        segs_out.append({
            "from": dep.get("iataCode"),
            "to":   arr.get("iataCode"),
            "dep_time": dep.get("at"),
            "arr_time": arr.get("at"),
            "carrier": carriers.get(code, code),
            "carrierCode": code,
            "number": seg.get("number"),
            "duration": seg.get("duration"),
        })
    return segs_out

# ---------- Main tool ----------
def search_flights(
    origin: str,
    destination: str,
    date_from: str,
    date_to: Optional[str] = None,
    nonstop_only: bool = False,
    cabin: Optional[str] = None,                 # ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
    max_price: Optional[float] = None,
    preferred_carriers: Optional[Iterable[str]] = None,  # ["BA","AA"]
    adults: int = 1,
    currency: str = "USD",
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search flights via Amadeus Flight Offers Search.

    For agents:
    - You may pass CITY/airport NAMES (e.g., "New York", "Heathrow") or IATA CODES ("NYC","JFK","LHR").
      This tool resolves names to IATA automatically.
    - Include user preferences if available:
        • nonstop_only: true/false
        • cabin: ECONOMY | PREMIUM_ECONOMY | BUSINESS | FIRST
        • max_price: numeric upper bound in the selected currency (e.g., 1200)
        • preferred_carriers: list like ["BA","AA"]
    - Always provide date_from (YYYY-MM-DD). date_to optional for return.

    Returns a list of offers (sorted by price) with fields:
      {
        "price": "1234.56",
        "price_num": 1234.56,
        "currency": "USD",
        "one_way": bool,
        "outbound": [ {from,to,dep_time,arr_time,carrier,carrierCode,number,duration}, ... ],
        "return":   [ ... ] or [],
        "stops_outbound": int,
        "stops_return": int
      }
    """
    # Debug to see what the agent passed
    print(f"[flight_api] origin={origin!r} dest={destination!r} "
          f"date_from={date_from} date_to={date_to} nonstop={nonstop_only} "
          f"cabin={cabin} max_price={max_price} carriers={preferred_carriers}")

    token = get_amadeus_access_token()

    # Resolve locations
    orig = resolve_loc_code(origin, token)
    dest = resolve_loc_code(destination, token)
    if not orig or not dest:
        print(f"[flight_api] Could not resolve codes: origin={origin!r}->{orig}, destination={destination!r}->{dest}")
        return []

    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "originLocationCode": orig,
        "destinationLocationCode": dest,
        "departureDate": date_from,
        "adults": adults,
        "currencyCode": currency,
        "max": max_results,
    }
    if date_to:
        params["returnDate"] = date_to
    if nonstop_only:
        params["nonStop"] = "true"
    if cabin:
        params["travelClass"] = cabin.upper()
    if max_price is not None:
        # Amadeus supports maxPrice as query param
        params["maxPrice"] = int(max_price)
    if preferred_carriers:
        # params["includedCarriers"] = ",".join([c.upper() for c in preferred_carriers])
        params["includedAirlineCodes"] = ",".join([c.upper() for c in preferred_carriers])

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 400:
        print("Amadeus API error:", resp.text)
    resp.raise_for_status()
    payload = resp.json()
    carriers = (payload.get("dictionaries", {}) or {}).get("carriers", {}) or {}

    out: List[Dict[str, Any]] = []
    for offer in payload.get("data", []) or []:
        price_str = (offer.get("price", {}) or {}).get("total")
        price_num = _num(price_str)
        its = offer.get("itineraries", []) or []

        outbound_segments: List[Dict[str, Any]] = []
        return_segments:  List[Dict[str, Any]] = []

        if len(its) >= 1:
            outbound_segments = _flatten_segments(its[0], carriers)
        if len(its) >= 2:
            return_segments  = _flatten_segments(its[1], carriers)

        out.append({
            "price": price_str,
            "price_num": price_num,
            "currency": (offer.get("price", {}) or {}).get("currency"),
            "one_way": len(its) < 2,
            "outbound": outbound_segments,
            "return": return_segments,
            "stops_outbound": max(0, len(outbound_segments) - 1) if outbound_segments else 0,
            "stops_return":  max(0, len(return_segments)  - 1) if return_segments else 0,
        })

    # sort by numeric price if present
    out.sort(key=lambda r: (r.get("price_num") is None, r.get("price_num", 0.0)))

    return out

# Local test
if __name__ == "__main__":
    if load_dotenv:
        load_dotenv()
    try:
        flights = search_flights(
            origin="Delhi", destination="Mumbai", date_from="2025-10-15", date_to="2025-10-17",
            nonstop_only=True, cabin="ECONOMY", max_price=1500,
        )
        print(f"Found {len(flights)} flights:")
        for f in flights[:5]:
            print(f)
    except Exception as e:
        print("Error:", e)
