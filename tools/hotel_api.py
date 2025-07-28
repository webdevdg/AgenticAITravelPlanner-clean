import os
import requests
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def get_amadeus_access_token() -> str:
    client_id = os.getenv("AMADEUS_CLIENT_ID")
    client_secret = os.getenv("AMADEUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET must be set in environment variables.")
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def get_hotel_ids(city_code: str, access_token: str) -> list:
    url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"cityCode": city_code}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    return [hotel["hotelId"] for hotel in data.get("data", [])]

def search_hotels(city: str, checkin: str, checkout: str) -> List[Dict[str, Any]]:
    """
    Search for hotels in a city using the Amadeus Hotel Search API.
    Args:
        city (str): IATA city code (e.g., 'NYC' for New York).
        checkin (str): Check-in date in 'YYYY-MM-DD' format.
        checkout (str): Check-out date in 'YYYY-MM-DD' format.
    Returns:
        List[Dict[str, Any]]: List of hotel options.
    """
    access_token = get_amadeus_access_token()
    hotel_ids = get_hotel_ids(city, access_token)
    if not hotel_ids:
        print("No hotels found for this city.")
        return []
    url = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "hotelIds": ",".join(hotel_ids[:20]),  # Amadeus may limit the number of IDs
        "checkInDate": checkin,
        "checkOutDate": checkout,
        "adults": 1,
        "roomQuantity": 1,
        "currency": "USD",
        "bestRateOnly": "true"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 400:
        print("Amadeus API error:", response.text)
    response.raise_for_status()
    data = response.json()
    hotels = data.get("data", [])
    results = []
    for hotel in hotels:
        hotel_info = hotel.get("hotel", {})
        offers = hotel.get("offers", [])
        for offer in offers:
            results.append({
                "name": hotel_info.get("name"),
                "address": hotel_info.get("address", {}).get("lines", []),
                "city": hotel_info.get("address", {}).get("cityName"),
                "price": offer.get("price", {}).get("total"),
                "currency": offer.get("price", {}).get("currency"),
                "checkInDate": offer.get("checkInDate"),
                "checkOutDate": offer.get("checkOutDate"),
                "room": offer.get("room", {}).get("typeEstimated", {}),
                "description": offer.get("room", {}).get("description", {}).get("text"),
                "bookingLink": offer.get("urls", {}).get("booking")
            })
    return results

if __name__ == "__main__":
    if load_dotenv:
        load_dotenv()
    else:
        print("[INFO] python-dotenv not installed, skipping .env loading.")

    try:
        hotels = search_hotels("NYC", "2025-08-10", "2025-08-12")
        print(f"Found {len(hotels)} hotels:")
        for h in hotels:
            print(h)
    except Exception as e:
        print(f"Error: {e}")
