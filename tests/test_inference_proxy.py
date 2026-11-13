import asyncio
from contextlib import asynccontextmanager
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
import pytest
from sqlalchemy import event as sqlalchemy_event
from starlette.middleware.base import BaseHTTPMiddleware

from gpustack import envs
from gpustack.api import exceptions
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.config.config import Config
from gpustack.gateway.utils import router_header_key
from gpustack.routes import openai as openai_routes
from gpustack.routes.worker import proxy as worker_proxy
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.server import db
from gpustack.server.services import ModelService
from gpustack.worker import worker as worker_module
from gpustack.worker import serve_manager as serve_module
from gpustack.worker.serve_manager import ServeManager

from test_watch import project_context, seed, live_server, eventually


SSE = 'event: message\ndata: {"text":\ndata: "你好"}\n\ndata: [DONE]\n\n'.encode()


@asynccontextmanager
async def proxy_context(tmp_path, monkeypatch):
    async with project_context(tmp_path, monkeypatch) as env:
        selected_model, first = await seed(env.client, state="running")
        engine = FastAPI()
        requests = []
        worker_requests = []
        release = asyncio.Event()
        disconnected = asyncio.Event()
        upload_started = asyncio.Event()

        @engine.post('/v1/{path:path}')
        async def infer(path: str, request: Request):
            if request.headers.get('x-upload-flow'):
                iterator = request.stream()
                first_chunk = await anext(iterator)
                upload_started.set()
                raw = first_chunk + b''.join([part async for part in iterator])
            else:
                raw = await request.body()
            body = json.loads(raw) if request.headers.get('content-type', '').startswith('application/json') else None
            form = None
            if request.headers.get('content-type', '').startswith('multipart/form-data'):
                parsed = await request.form()
                form = [(key, await value.read() if hasattr(value, 'read') else value) for key, value in parsed.multi_items()]
                await parsed.close()
            requests.append(SimpleNamespace(path=path, port=request.url.port, headers=request.headers, body=body, form=form,
                                            query=list(request.query_params.multi_items()), raw=raw))
            mode = (body or {}).get('mode')
            if mode == 'slow':
                await asyncio.sleep(.3)
            if mode == 'empty':
                return Response(status_code=204)
            if mode == 'error':
                return JSONResponse({'error': {'message': 'fixture invalid', 'type': 'BadRequest', 'code': 422}}, status_code=422)
            if body and body.get('stream'):
                async def chunks():
                    try:
                        yield SSE[:11]
                        await release.wait()
                        yield SSE[11:43]
                        yield SSE[43:]
                    finally:
                        disconnected.set()
                return StreamingResponse(chunks(), media_type='text/event-stream')
            data = b'\x00\xfffixture audio' if path == 'audio/speech' else json.dumps({'port': request.url.port, 'path': path}).encode()
            response = Response(gzip.compress(data), headers={'Content-Encoding': 'gzip'}, media_type='audio/wav' if path == 'audio/speech' else 'application/json')
            response.headers.append('set-cookie', 'first=1')
            response.headers.append('set-cookie', 'second=2')
            return response

        cfg = Config(data_dir=str(tmp_path/'worker'), token='registration-fixture', worker_port=10150)
        Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.data_dir)/'worker_token').write_text('dedicated-fixture')
        manager = ServeManager(lambda: 1, lambda: None, cfg)
        worker = object.__new__(worker_module.Worker)
        worker._config, worker._serve_manager = cfg, manager
        worker._worker_ip, worker._address, worker._is_embedded = '127.0.0.1', '127.0.0.1', False
        captured = []
        class CaptureServer:
            def __init__(self, config):
                captured.append(config.app)
            async def serve(self):
                pass
        with monkeypatch.context() as capture:
            capture.setattr(worker_module.uvicorn, 'Server', CaptureServer)
            await worker._serve_apis()
        worker_app = captured[0]
        async def observe_worker(request, call_next):
            worker_requests.append((request.headers.get('authorization'), request.headers.get(router_header_key)))
            return await call_next(request)
        worker_app.add_middleware(BaseHTTPMiddleware, dispatch=observe_worker)

        async with live_server(engine) as first_engine, live_server(engine) as second_engine:
            async with worker_app.router.lifespan_context(worker_app), live_server(worker_app) as worker_url:
                async with db.async_session() as session:
                    from gpustack.schemas.workers import Worker
                    registered = await Worker.create(session, {'name': 'proxy-node', 'hostname': 'proxy-node', 'ip': '127.0.0.1', 'ifname': 'lo0', 'status': None,
                        'port': urlparse(worker_url).port, 'worker_uuid': 'fixture-worker', 'state': 'ready',
                        'token': 'dedicated-fixture', 'cluster_id': selected_model['cluster_id']})
                    model = await Model.one_by_id(session, selected_model['id'])
                    await ModelService(session).update(model, {'ready_replicas': 2})
                rows = []
                for index, engine_url in enumerate((first_engine, second_engine)):
                    payload = {**first, 'worker_id': registered.id, 'worker_ip': '127.0.0.1', 'port': urlparse(engine_url).port,
                               'ports': [urlparse(engine_url).port], 'state': 'running'}
                    if index == 0:
                        response = await env.client.put(f"/v2/model-instances/{first['id']}", json=payload)
                    else:
                        payload.update(name='second-instance')
                        payload.pop('id')
                        response = await env.client.post('/v2/model-instances', json=payload)
                    assert response.status_code == 200, response.text
                    rows.append(response.json())
                manager._model_instance_by_instance_id = {row['id']: ModelInstance.model_validate(row) for row in rows}
                manager._worker_id_getter = lambda: registered.id
                async with live_server(env.app) as server_url:
                    async with httpx.AsyncClient(base_url=server_url, timeout=5, trust_env=False) as client:
                        yield SimpleNamespace(**vars(env), network=client, requests=requests, worker_requests=worker_requests,
                            model=selected_model, rows=rows, manager=manager, worker_app=worker_app, worker_url=worker_url,
                            ports=[urlparse(first_engine).port, urlparse(second_engine).port], release=release, disconnected=disconnected,
                            upload_started=upload_started)


def test_two_hop_round_robin_models_alias_and_cache_invalidation(tmp_path, monkeypatch):
    async def check():
        async with proxy_context(tmp_path, monkeypatch) as env:
            models = (await env.network.get('/v1/models')).json()
            assert models['object'] == 'list' and [row['id'] for row in models['data']] == [env.model['name']]
            ports = []
            for index in range(4):
                response = await env.network.post('/v1/chat/completions' if index < 3 else '/v1-openai/chat/completions',
                    json={'model': env.model['name']}, headers={'Authorization': 'Bearer user-secret', router_header_key: 'spoofed'})
                assert response.status_code == 200, response.text
                ports.append(response.json()['port'])
                assert response.headers.get_list('set-cookie') == ['first=1', 'second=2']
                assert 'content-encoding' not in response.headers
            assert ports == env.ports * 2
            assert all(auth == 'Bearer dedicated-fixture' and route.startswith('model-') for auth, route in env.worker_requests)
            assert all('authorization' not in row.headers and router_header_key not in row.headers for row in env.requests)
            assert all(row.path == 'chat/completions' for row in env.requests)
            row = env.rows[0]
            response = await env.client.put(f"/v2/model-instances/{row['id']}", json={**row, 'state': 'error'})
            assert response.status_code == 200
            response = await env.network.post('/v1/chat/completions', json={'model': env.model['name']})
            assert response.json()['port'] == env.ports[1]
    asyncio.run(check())


def test_error_formats_and_worker_authentication(tmp_path, monkeypatch):
    async def check():
        async with proxy_context(tmp_path, monkeypatch) as env:
            for body, code in [({'model': 'no-such'}, 404), ({}, 400), ({'model': 3}, 400), ({'model': env.model['name'], 'stream': 'false'}, 400)]:
                response = await env.network.post('/v1/chat/completions', json=body)
                assert response.status_code == code and response.json()['error']['code'] == code
            response = await env.network.post('/v1/chat/completions', content=b'{', headers={'content-type': 'application/json'})
            assert response.status_code == 400
            # OpenAI errors must not change the format of later management errors.
            response = await env.client.get('/v2/models/999999')
            assert response.status_code == 404 and 'reason' in response.json() and 'error' not in response.json()
            async with httpx.AsyncClient(base_url=env.worker_url, trust_env=False) as worker:
                route = {router_header_key: f"model-{env.model['id']}-{env.rows[0]['id']}.static"}
                for headers in ({}, route, {**route, 'Authorization': 'Bearer wrong'}):
                    response = await worker.post('/v1/chat/completions', json={}, headers=headers)
                    assert response.status_code == 401
                for token in ('dedicated-fixture', 'registration-fixture'):
                    response = await worker.post('/v1/chat/completions', json={}, headers={**route, 'Authorization': 'Bearer '+token})
                    assert response.status_code == 200
                response = await worker.post('/v1/chat/completions', headers={'Authorization': 'Bearer dedicated-fixture'})
                assert response.status_code == 400
                response = await worker.post('/v1/chat/completions', headers={'Authorization': 'Bearer dedicated-fixture', router_header_key: 'invalid'})
                assert response.status_code == 404
                assert (await worker.get('/healthz')).status_code == 200
            for row in env.rows:
                assert (await env.client.put(f"/v2/model-instances/{row['id']}", json={**row, 'state': 'starting'})).status_code == 200
            response = await env.network.post('/v1/chat/completions', json={'model': env.model['name']})
            assert response.status_code == 503 and response.json()['error']['message'] == 'No running instances available'
    asyncio.run(check())


def test_upstream_status_binary_query_and_multipart(tmp_path, monkeypatch):
    async def check():
        async with proxy_context(tmp_path, monkeypatch) as env:
            for stream in (False, True):
                response = await env.network.post('/v1/chat/completions', json={'model': env.model['name'], 'mode': 'error', 'stream': stream})
                assert response.status_code == 422 and response.json()['error']['message'] == 'fixture invalid'
            response = await env.network.post('/v1/audio/speech?x=1&x=2', json={'model': env.model['name']})
            assert response.content == b'\x00\xfffixture audio' and response.headers['content-type'] == 'audio/wav'
            assert env.requests[-1].query == [('x', '1'), ('x', '2')]
            response = await env.network.post('/v1/audio/transcriptions', files=[('model', (None, env.model['name'])),
                ('stream', (None, 'false')), ('language', (None, 'zh')), ('language', (None, 'en')),
                ('file', ('voice.wav', b'fixture-audio', 'audio/wav'))])
            assert response.status_code == 200
            form = env.requests[-1].form
            assert ('file', b'fixture-audio') in form and [value for key, value in form if key == 'language'] == ['zh', 'en']
            for stream in (False, True):
                response = await env.network.post('/v1/chat/completions', json={'model': env.model['name'], 'mode': 'empty', 'stream': stream})
                assert response.status_code == 204 and response.content == b''
            monkeypatch.setattr(envs, 'PROXY_TIMEOUT', .05)
            for stream in (False, True):
                response = await env.network.post('/v1/chat/completions', json={'model': env.model['name'], 'mode': 'slow', 'stream': stream})
                assert response.status_code == 504 and response.json()['error']['code'] == 504
    asyncio.run(check())


def test_sse_arrives_before_completion_and_releases_database_session(tmp_path, monkeypatch):
    async def check():
        async with proxy_context(tmp_path, monkeypatch) as env:
            active = set()
            @sqlalchemy_event.listens_for(db.engine.sync_engine, 'checkout')
            def checkout(connection, record, proxy):
                active.add(id(connection))
            @sqlalchemy_event.listens_for(db.engine.sync_engine, 'checkin')
            def checkin(connection, record):
                active.discard(id(connection))
            async with env.network.stream('POST', '/v1/chat/completions', json={'model': env.model['name'], 'stream': True}) as response:
                assert response.status_code == 200
                iterator = response.aiter_bytes()
                first = await asyncio.wait_for(anext(iterator), timeout=2)
                assert first == SSE[:11] and not env.disconnected.is_set()
                assert not active, 'Streaming inference retained a database connection'
                assert env.requests[-1].body['stream_options'] == {'include_usage': True}
                env.release.set()
                assert first + b''.join([part async for part in iterator]) == SSE
            assert env.manager._last_successful_inference
    asyncio.run(check())


def test_stream_disconnect_closes_upstream(tmp_path, monkeypatch):
    async def check():
        async with proxy_context(tmp_path, monkeypatch) as env:
            async with env.network.stream('POST', '/v1/chat/completions', json={'model': env.model['name'], 'stream': True}) as response:
                assert await anext(response.aiter_bytes()) == SSE[:11]
            await eventually(env.disconnected.is_set, timeout=3)
    asyncio.run(check())


def test_worker_forwards_chunked_upload_before_body_finishes(tmp_path, monkeypatch):
    async def check():
        async with proxy_context(tmp_path, monkeypatch) as env:
            row = env.rows[0]
            body = json.dumps({'model': env.model['name']}).encode()
            async def chunks():
                yield body[:6]
                await asyncio.wait_for(env.upload_started.wait(), timeout=2)
                yield body[6:]
            async with httpx.AsyncClient(base_url=env.worker_url, trust_env=False) as client:
                response = await client.post('/v1/chat/completions', content=chunks(), headers={
                    'Authorization': 'Bearer dedicated-fixture', 'Content-Type': 'application/json', 'X-Upload-Flow': 'true',
                    router_header_key: f"model-{env.model['id']}-{row['id']}.static"})
                assert response.status_code == 200
                assert env.requests[-1].raw == body
    asyncio.run(check())


@pytest.mark.parametrize('error,expected', [(aiohttp.ClientConnectionError, 503), (asyncio.TimeoutError, 504), (RuntimeError, 500)])
@pytest.mark.parametrize('mid_stream', [False, True])
def test_stream_errors_have_one_status_and_close_upstream(monkeypatch, error, expected, mid_stream):
    async def check():
        closed = []
        async def stream(**kwargs):
            try:
                if mid_stream:
                    yield b'data: {"ok":true}\n\n', {'Content-Type': 'text/event-stream'}, 200
                raise error('fixture failure')
            finally:
                closed.append(True)
        monkeypatch.setattr(openai_routes, 'stream_to_worker', stream)
        upstream = openai_routes._stream_response(SimpleNamespace(id=1), 'POST', 'v1/chat/completions', {}, b'{}', None, None)
        response = StreamingResponseWithStatusCode(upstream, media_type='text/event-stream')
        events = []
        async def send(event):
            events.append(event)
        await response.stream_response(send)
        starts = [event for event in events if event['type'] == 'http.response.start']
        assert len(starts) == 1 and starts[0]['status'] == (200 if mid_stream else expected)
        data = b''.join(event.get('body', b'') for event in events)
        if mid_stream:
            assert b'data: {"error":' in data
        else:
            assert json.loads(data)['error']['code'] == expected
            assert (b'content-type', b'application/json') in starts[0]['headers']
        assert closed == [True]
    asyncio.run(check())


def test_port_lookup_and_recent_success_skip_active_health_probe(tmp_path, monkeypatch):
    manager = ServeManager(lambda: 1, lambda: None, Config(data_dir=str(tmp_path)))
    mi = ModelInstance.model_validate({'id': 1, 'name': 'instance', 'model_id': 1, 'model_name': 'qwen',
        'source': 'model_scope', 'model_scope_model_id': 'Qwen/Qwen3-0.6B', 'state': 'running', 'worker_id': 1,
        'port': 40001, 'ports': None})
    manager._model_instance_by_instance_id[1] = mi
    assert manager.get_instance_port_by_model_instance_id(1) == 40001
    assert manager.get_instance_port_by_model_instance_id(999) is None
    model = Model.model_validate({'name': 'qwen', 'source': 'model_scope', 'model_scope_model_id': 'Qwen/Qwen3-0.6B', 'distributed_inference_across_workers': True,
        'env': {'GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED': 'true', 'GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL': '30'}})
    manager._get_model = lambda mi: model
    manager._is_provisioning = lambda mi: False
    monkeypatch.setattr(serve_module, 'is_inference_ready', lambda *args, **kwargs: pytest.fail('Recent traffic must skip active probing'))
    manager.record_successful_inference(1)
    manager.sync_model_instances_inference_health()
    mi.state = ModelInstanceStateEnum.ERROR
    assert manager.get_instance_port_by_model_instance_id(1) is None
