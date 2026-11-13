"""两个代理跳点共用的响应头过滤规则。"""

EXCLUDED_RESPONSE_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade",
    "server", "date", "content-length", "content-encoding",
})


def filter_response_headers(headers):
    # aiohttp 已解压正文；长度和传输方式由当前 HTTP 服务重新生成。
    # 保留列表形式，使 Set-Cookie 等重复响应头能够经过两跳。
    items = list(headers.items())
    excluded = set(EXCLUDED_RESPONSE_HEADERS)
    for key, value in items:
        if key.lower() == "connection":
            excluded.update(part.strip().lower() for part in value.split(","))
    return [(key, value) for key, value in items if key.lower() not in excluded]
