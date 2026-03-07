import asyncio
import unittest

from app.services.http_clients import close_http_clients, get_oauth_http_client


class HttpClientsTests(unittest.TestCase):
    def tearDown(self):
        asyncio.run(close_http_clients())

    def test_oauth_client_is_reused_and_closed(self):
        first = get_oauth_http_client()
        second = get_oauth_http_client()
        self.assertIs(first, second)
        self.assertFalse(first.is_closed)

        asyncio.run(close_http_clients())
        self.assertTrue(first.is_closed)

        third = get_oauth_http_client()
        self.assertIsNot(first, third)
        self.assertFalse(third.is_closed)


if __name__ == "__main__":
    unittest.main()
