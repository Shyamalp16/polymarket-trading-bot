import requests

from ..exceptions import RelayerApiException

GET = "GET"
POST = "POST"
DELETE = "DELETE"
PUT = "PUT"


def request(endpoint: str, method: str, headers=None, data=None):
    try:
        resp = requests.request(
            method=method, url=endpoint, headers=headers, json=data if data else None
        )
        if resp.status_code != 200:
            raise RelayerApiException(resp)

        try:
            return resp.json()
        except requests.JSONDecodeError:
            return resp.text

    except requests.RequestException:
        raise RelayerApiException(error_msg="Request exception!")


def post(endpoint, headers=None, data=None):
    return request(endpoint, POST, headers, data)


def get(endpoint, headers=None, data=None):
    return request(endpoint, GET, headers, data)


def delete(endpoint, headers=None, data=None):
    return request(endpoint, DELETE, headers, data)
