import os
import requests
from typing import List, Dict, Any

# Testing
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def get_amadeus_access_token() -> str:
    """
    Authenticate with Amadeus and return an access token.
    """
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

def search_flights(origin: str, destination: str, date_from: str, date_to: str = None) -> List[Dict[str, Any]]:
    """
    Search for flights using the Amadeus Flight Offers Search API.
    Args:
        origin (str): IATA code of the origin airport/city.
        destination (str): IATA code of the destination airport/city.
        date_from (str): Departure date in format 'YYYY-MM-DD'.
        date_to (str, optional): Return date in format 'YYYY-MM-DD'.
    Returns:
        List[Dict[str, Any]]: List of flight options.
    """
    access_token = get_amadeus_access_token()
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": date_from,
        "adults": 1,
        "currencyCode": "USD",
        "max": 10
    }
    if date_to:
        params["returnDate"] = date_to
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    offers = data.get("data", [])
    results = []
    for offer in offers:
        itineraries = offer.get("itineraries", [])
        price = offer.get("price", {}).get("total")
        segments = []
        for itinerary in itineraries:
            for segment in itinerary.get("segments", []):
                segments.append({
                    "departure": segment["departure"],
                    "arrival": segment["arrival"],
                    "carrierCode": segment["carrierCode"],
                    "number": segment["number"]
                })
        results.append({
            "price": price,
            "segments": segments
        })
    return results

# To test and run from this file
if __name__ == "__main__":
    if load_dotenv:
        load_dotenv()
    else:
        print("[INFO] python-dotenv not installed, skipping .env loading.")
    # Example test: JFK to LHR, July 1, 2024 (one-way)
    try:
        # flights = search_flights("JFK", "LHR", "2025-08-10")
        flights = search_flights("JFK", "LHR", "2025-08-10", "2025-08-20")
        print(f"Found {len(flights)} flights:")
        for f in flights:
            print(f)
    except Exception as e:
        print(f"Error: {e}")
