# tools/hotel_api.py
import os
import time
import requests
from typing import List, Dict, Any, Optional
from datetime import date, datetime, timedelta

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ------------------ Amadeus token cache ------------------
_TOKEN: Dict[str, Any] = {"access_token": None, "expires_at": 0.0}

def get_amadeus_access_token() -> str:
    """Get (and cache) an Amadeus OAuth token. Reuses until expiry."""
    now = time.time()
    if _TOKEN["access_token"] and now < _TOKEN["expires_at"]:
        return _TOKEN["access_token"]

    client_id = os.getenv("AMADEUS_CLIENT_ID")
    client_secret = os.getenv("AMADEUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET must be set.")

    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    resp = requests.post(url, data=data, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    _TOKEN["access_token"] = payload["access_token"]
    _TOKEN["expires_at"] = now + float(payload.get("expires_in", 1800)) - 60  # refresh 1m early
    return _TOKEN["access_token"]

# ------------------ Helpers ------------------
def resolve_city_code(city_or_code: str, access_token: str) -> Optional[str]:
    """Accepts city NAME ('New York') or IATA CODE ('NYC'). Returns 3-letter IATA code."""
    c = (city_or_code or "").strip()
    if len(c) == 3 and c.isalpha():
        return c.upper()
    url = "https://test.api.amadeus.com/v1/reference-data/locations"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"subType": "CITY", "keyword": c}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    return data[0].get("iataCode") if data else None

def get_hotel_ids(city_code: str, access_token: str) -> list:
    url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"cityCode": city_code}
    response = requests.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    return [hotel["hotelId"] for hotel in data.get("data", [])]

def _ensure_future_dates(checkin: str, checkout: str) -> tuple[str, str]:
    today = date.today()
    try:
        ci = datetime.fromisoformat(checkin).date()
    except Exception:
        ci = today + timedelta(days=7)
    try:
        co = datetime.fromisoformat(checkout).date()
    except Exception:
        co = ci + timedelta(days=2)
    if ci <= today:
        ci = today + timedelta(days=7)
    if co <= ci:
        co = ci + timedelta(days=2)
    return ci.isoformat(), co.isoformat()

# ------------------ Main tool ------------------
def search_hotels(
    city: str,
    checkin: str,
    checkout: str,
    hotel_class: Optional[str] = None,
    max_price: Optional[float] = None,
    adults: int = 1,
    room_quantity: int = 1,
    currency: str = "USD",
) -> List[Dict[str, Any]]:
    """
    Find hotels for a city and date range using the Amadeus Hotel Offers API.

    For agents:
    - You may pass a CITY NAME (e.g., "New York") or a CITY IATA CODE (e.g., "NYC").
      This tool will resolve names to IATA codes automatically.
    - ALWAYS include known user preferences if available:
        • hotel_class: e.g., "4-star" / "5-star"
        • max_price: numeric upper bound in the given currency (e.g., 2000)
    - adults, room_quantity, currency are optional.

    Returns a list of hotels with fields:
      name, address, city, stars (if provided by API), price, price_num, currency,
      checkInDate, checkOutDate, room, description, bookingLink
    """
    print(f"[hotel_api] city={city!r} checkin={checkin} checkout={checkout} "
          f"hotel_class={hotel_class} max_price={max_price}")

    checkin, checkout = _ensure_future_dates(checkin, checkout)
    token = get_amadeus_access_token()

    city_code = resolve_city_code(city, token)
    if not city_code:
        print(f"[hotel_api] Could not resolve city code for {city!r}")
        return []

    hotel_ids = get_hotel_ids(city_code, token)
    if not hotel_ids:
        print("[hotel_api] No hotel IDs found for this city.")
        return []

    url = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "hotelIds": ",".join(hotel_ids[:20]),
        "checkInDate": checkin,
        "checkOutDate": checkout,
        "adults": adults,
        "roomQuantity": room_quantity,
        "currency": currency,
        "bestRateOnly": "true",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 400:
        print("Amadeus API error:", resp.text)
    resp.raise_for_status()

    results: List[Dict[str, Any]] = []
    for item in resp.json().get("data", []):
        h = item.get("hotel", {}) or {}
        rating = h.get("rating")  # often a string like "4" or "5"
        offers = item.get("offers", []) or []
        for off in offers:
            price_str = (off.get("price", {}) or {}).get("total")
            price_num = None
            try:
                price_num = float(price_str) if price_str is not None else None
            except Exception:
                pass

            results.append({
                "name": h.get("name"),
                "address": (h.get("address", {}) or {}).get("lines", []),
                "city": (h.get("address", {}) or {}).get("cityName"),
                "stars": int(rating) if rating and str(rating).isdigit() else None,
                "price": price_str,
                "price_num": price_num,
                "currency": (off.get("price", {}) or {}).get("currency"),
                "checkInDate": off.get("checkInDate"),
                "checkOutDate": off.get("checkOutDate"),
                "room": (off.get("room", {}) or {}).get("typeEstimated", {}),
                "description": (off.get("room", {}) or {}).get("description", {}).get("text"),
                "bookingLink": (off.get("urls", {}) or {}).get("booking"),
            })

    # --------- apply optional filters from prefs ---------
    if max_price is not None:
        results = [r for r in results if (r.get("price_num") is not None and r["price_num"] <= float(max_price))]
    if hotel_class:
        # accept "4-star" format or just "4"
        want = "".join(ch for ch in hotel_class if ch.isdigit())
        if want:
            results = [r for r in results if r.get("stars") and str(r["stars"]) == want]

    # sort by price if available
    results.sort(key=lambda r: (r.get("price_num") is None, r.get("price_num", 0.0)))

    return results

if __name__ == "__main__":
    if load_dotenv:
        load_dotenv()
    try:
        hotels = search_hotels("New York", "2025-08-10", "2025-08-12", hotel_class="4-star", max_price=2000)
        print(f"Found {len(hotels)} hotels:")
        for h in hotels[:5]:
            print(h)
    except Exception as e:
        print(f"Error: {e}")
