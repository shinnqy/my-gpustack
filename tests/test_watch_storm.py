"""Run the chapter-seven watch checkpoint against an already running Server."""

import argparse
import asyncio
import json

import httpx


async def one(client):
    async with client.stream(
        "GET", "/v2/model-instances", params={"watch": "true"}, timeout=20,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line:
                return json.loads(line)
    raise ConnectionError("Watch closed before sending its initial snapshot")


async def main(server_url, watchers):
    async with httpx.AsyncClient(base_url=server_url) as client:
        events = await asyncio.wait_for(
            asyncio.gather(*(one(client) for _ in range(watchers))), timeout=30,
        )
    print(f"{len(events)} streams got the snapshot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:30080")
    parser.add_argument("--watchers", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main(args.server_url, args.watchers))
