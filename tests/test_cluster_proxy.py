import asyncio
from contextlib import asynccontextmanager
import gzip
import json
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
from types import SimpleNamespace
from urllib.parse import urlparse

from aiohttp import web
import httpx
import pytest
from sqlalchemy import event as sqlalchemy_event
from starlette.datastructures import Headers

from gpustack.config.config import Config
from gpustack.gpu_instances import gateway_client
from gpustack.routes.worker import cluster_proxy
from gpustack.schemas.workers import Worker
from gpustack.server import db
from gpustack.worker import worker as worker_module
from gpustack.worker.serve_manager import ServeManager

from test_watch import project_context, live_server, eventually
from test_resource_fit import worker


@pytest.fixture(scope='module')
def certificates():
    with tempfile.TemporaryDirectory(prefix='kube-tls-',dir='/tmp') as temp:
        root=Path(temp)
        config=root/'openssl.cnf'
        config.write_text('[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ext\n[dn]\nCN=localhost\n[ext]\nsubjectAltName=DNS:localhost,IP:127.0.0.1\nbasicConstraints=critical,CA:true\nkeyUsage=critical,digitalSignature,keyEncipherment,keyCertSign\n')
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-config',str(config),
            '-keyout',str(root/'key.pem'),'-out',str(root/'cert.pem')],check=True,capture_output=True)
        yield root/'cert.pem',root/'key.pem'


@asynccontextmanager
async def kube_proxy_context(tmp_path,monkeypatch,certificates):
    async with project_context(tmp_path,monkeypatch,database_file=True) as env:
        cert,key=certificates
        records=[];worker_headers=[];release=asyncio.Event();closed=asyncio.Event();break_stream=asyncio.Event()
        async def handle(request):
            body=await request.read()
            records.append(SimpleNamespace(method=request.method,path=request.path,query=list(request.query.items()),headers=request.headers,body=body))
            if request.path.endswith('/broken'):
                response=web.StreamResponse(headers={'Content-Type':'application/json'})
                await response.prepare(request)
                await response.write(b'{"type":"ADDED"}\n')
                await break_stream.wait()
                request.transport.close()
                return response
            if request.path.endswith('/watch'):
                response=web.StreamResponse(headers={'Content-Type':'application/json'})
                await response.prepare(request)
                try:
                    await response.write(b'{"type":"ADDED"}\n')
                    while not release.is_set():
                        await asyncio.sleep(.02)
                        await response.write(b'\n')
                    await response.write(b'{"type":"MODIFIED"}\n')
                except (ConnectionError,asyncio.CancelledError):pass
                finally:closed.set()
                return response
            if request.path.endswith('/redirect'):
                return web.Response(status=302,headers={'Location':'/redirected'})
            if request.path.endswith('/error'):
                return web.json_response({'kind':'Status','message':'fixture forbidden'},status=403)
            if request.path.endswith('/binary'):
                return web.Response(body=b'\x00\xffprotobuf-fixture',headers={'Content-Type':'application/vnd.kubernetes.protobuf'})
            payload=json.dumps({'kind':'NodeList','items':[{'metadata':{'name':'fixture-node'}}]}).encode()
            return web.Response(body=gzip.compress(payload),headers=[('Content-Type','application/json'),('Content-Encoding','gzip'),
                ('Set-Cookie','first=1'),('Set-Cookie','second=2')])
        kube=web.Application();kube.router.add_route('*','/{path:.*}',handle)
        runner=web.AppRunner(kube);await runner.setup()
        tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(cert,key)
        site=web.TCPSite(runner,'127.0.0.1',0,ssl_context=tls);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        token=tmp_path/'service-account-token';token.write_text('sa-first')
        monkeypatch.setenv('KUBERNETES_SERVICE_HOST','127.0.0.1')
        monkeypatch.setenv('KUBERNETES_SERVICE_PORT_HTTPS',str(port))
        monkeypatch.setattr(cluster_proxy,'TOKEN_PATH',token)
        monkeypatch.setattr(cluster_proxy,'CA_PATH',cert)
        monkeypatch.setattr(cluster_proxy,'_session_lock',asyncio.Lock())
        cluster_proxy._resolve_kube_target.cache_clear()
        config=Config(data_dir=str(tmp_path/'worker'),token='registration-fixture')
        Path(config.data_dir).mkdir(exist_ok=True,parents=True)
        (Path(config.data_dir)/'worker_token').write_text('dedicated-fixture')
        selected=object.__new__(worker_module.Worker)
        selected._config=config;selected._worker_ip='127.0.0.1';selected._address='127.0.0.1';selected._is_embedded=False
        selected._serve_manager=ServeManager(lambda:1,lambda:None,config)
        captured=[]
        class Capture:
            def __init__(self,config):captured.append(config.app)
            async def serve(self):pass
        with monkeypatch.context() as capture:
            capture.setattr(worker_module.uvicorn,'Server',Capture)
            await selected._serve_apis()
        worker_app=captured[0]
        class Observe:
            def __init__(self,app):self.app=app
            async def __call__(self,scope,receive,send):
                if scope['type']=='http':worker_headers.append(Headers(scope=scope))
                await self.app(scope,receive,send)
        worker_app.add_middleware(Observe)
        session=None
        try:
            async with worker_app.router.lifespan_context(worker_app),live_server(worker_app) as worker_url:
                cluster=(await env.client.post('/v2/clusters',json={'name':'kube','provider':'Kubernetes'})).json()
                async with db.async_session() as database:
                    node=await Worker.create(database,worker(1,cluster_id=cluster['id'],port=urlparse(worker_url).port,token='dedicated-fixture',ip='127.0.0.1'))
                async with live_server(env.app) as server_url:
                    async with httpx.AsyncClient(base_url=server_url,trust_env=False,timeout=5) as client:
                        yield SimpleNamespace(**vars(env),network=client,cluster=cluster,node=node,worker_url=worker_url,
                            worker_app=worker_app,records=records,worker_headers=worker_headers,token=token,release=release,closed=closed,break_stream=break_stream)
                session=getattr(worker_app.state,'kube_api_session',None)
            if session is not None:assert session.closed
        finally:
            release.set()
            break_stream.set()
            await runner.cleanup()
            cluster_proxy._resolve_kube_target.cache_clear()


@pytest.mark.parametrize('method',['GET','POST','PUT','PATCH','DELETE','HEAD','OPTIONS'])
def test_kubernetes_proxy_verbs_headers_query_and_binary(tmp_path,monkeypatch,certificates,method):
    async def check():
        async with kube_proxy_context(tmp_path,monkeypatch,certificates) as env:
            path=f"/v2/clusters/{env.cluster['id']}/proxy/api/v1/nodes?labelSelector=a&labelSelector=b"
            response=await env.network.request(method,path,content=b'{"fixture":true}' if method not in {'GET','HEAD','OPTIONS'} else None,
                headers={'Authorization':'Bearer caller-fixture','X-Api-Key':'private-fixture','Cookie':'private=1',
                    'X-GPUStack-Model-Instance':'bad-route','Content-Type':'application/json','Connection':'x-local-hop','X-Local-Hop':'private'})
            assert response.status_code==200,response.text
            upstream=env.records[-1]
            assert upstream.method==method and upstream.path=='/api/v1/nodes'
            assert upstream.headers['Authorization']=='Bearer sa-first'
            assert 'cookie' not in upstream.headers and 'x-api-key' not in upstream.headers
            assert 'x-gpustack-model-instance' not in upstream.headers
            assert 'x-local-hop' not in upstream.headers
            assert upstream.query==[('labelSelector','a'),('labelSelector','b')]
            assert env.worker_headers[-1]['Authorization']=='Bearer dedicated-fixture'
            if method not in {'GET','HEAD','OPTIONS'}:assert upstream.body==b'{"fixture":true}'
            if method!='HEAD':
                assert response.json()['kind']=='NodeList'
                assert response.headers.get_list('set-cookie')==['first=1','second=2']
                assert 'content-encoding' not in response.headers
    asyncio.run(check())


def test_proxy_preserves_errors_redirects_binary_and_rotated_sa_token(tmp_path,monkeypatch,certificates):
    async def check():
        async with kube_proxy_context(tmp_path,monkeypatch,certificates) as env:
            prefix=f"/v2/clusters/{env.cluster['id']}/proxy/api/v1"
            response=await env.network.get(prefix+'/error')
            assert response.status_code==403 and response.json()['kind']=='Status'
            response=await env.network.get(prefix+'/redirect')
            assert response.status_code==302 and response.headers['location']=='/redirected'
            assert all(record.path!='/redirected' for record in env.records)
            response=await env.network.get(prefix+'/binary')
            assert response.content==b'\x00\xffprotobuf-fixture'
            env.token.write_text('sa-rotated')
            assert (await env.network.get(prefix+'/nodes')).status_code==200
            assert env.records[-1].headers['Authorization']=='Bearer sa-rotated'
            async with httpx.AsyncClient(base_url=env.worker_url,trust_env=False) as client:
                assert (await client.get('/cluster-proxy/api/v1/nodes')).status_code==401
                assert (await client.get('/cluster-proxy/api/v1/nodes',headers={'Authorization':'Bearer wrong'})).status_code==401
    asyncio.run(check())


def test_proxy_watch_stream_releases_database_and_closes_upstream(tmp_path,monkeypatch,certificates):
    async def check():
        async with kube_proxy_context(tmp_path,monkeypatch,certificates) as env:
            active=set()
            @sqlalchemy_event.listens_for(db.engine.sync_engine,'checkout')
            def checkout(connection,record,proxy):active.add(id(connection))
            @sqlalchemy_event.listens_for(db.engine.sync_engine,'checkin')
            def checkin(connection,record):active.discard(id(connection))
            async with env.network.stream('GET',f"/v2/clusters/{env.cluster['id']}/proxy/api/v1/watch?watch=true") as response:
                first=await asyncio.wait_for(anext(response.aiter_bytes()),timeout=2)
                assert b'ADDED' in first and not active
            await eventually(env.closed.is_set,timeout=3)
    asyncio.run(check())


def test_proxy_provider_and_readiness_are_cluster_scoped(tmp_path,monkeypatch,certificates):
    async def check():
        async with kube_proxy_context(tmp_path,monkeypatch,certificates) as env:
            docker=(await env.client.post('/v2/clusters',json={'name':'docker'})).json()
            assert (await env.network.get(f"/v2/clusters/{docker['id']}/proxy/api/v1/nodes")).status_code==422
            assert (await env.network.get('/v2/clusters/999/proxy/api/v1/nodes')).status_code==404
            async with db.async_session() as session:
                node=await Worker.one_by_id(session,env.node.id)
                await node.update(session,{'state':'not_ready'})
                await Worker.create(session,worker(2,cluster_id=docker['id'],port=env.node.port,token='dedicated-fixture',ip='127.0.0.1'))
            assert (await env.network.get(f"/v2/clusters/{env.cluster['id']}/proxy/api/v1/nodes")).status_code==503
            assert not env.records
    asyncio.run(check())


def test_gateway_client_uses_tls_over_unix_socket(certificates):
    async def check():
        cert,key=certificates
        seen=[]
        async def handle(request):
            seen.append((request.method,list(request.query.items()),await request.json() if request.method=='POST' else None))
            return web.Response(status=204)
        app=web.Application();app.router.add_route('*','/apis/workers',handle)
        runner=web.AppRunner(app);await runner.setup()
        tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(cert,key)
        with tempfile.TemporaryDirectory(prefix='wg-test-',dir='/tmp') as temporary:
            path=str(Path(temporary)/'gateway.sock')
            site=web.UnixSite(runner,path,ssl_context=tls);await site.start()
            gateway_client.set_gateway_unix_path(path)
            try:
                await gateway_client.subscribe_worker('7','fixture-token',gvk=[('worker.gpustack.ai','v1','Instance')],force=True)
                await gateway_client.unsubscribe_worker('7')
                assert seen[0][0]=='POST' and ('cluster','7') in seen[0][1] and ('force','true') in seen[0][1]
                assert seen[0][2]=={'gvks':['worker.gpustack.ai/v1/Instance']}
                assert seen[1][0]=='DELETE'
            finally:
                gateway_client.set_gateway_unix_path(None)
                await runner.cleanup()
    asyncio.run(check())


def test_broken_kubernetes_stream_signals_transport_failure(tmp_path,monkeypatch,certificates):
    async def check():
        async with kube_proxy_context(tmp_path,monkeypatch,certificates) as env:
            async with env.network.stream('GET',f"/v2/clusters/{env.cluster['id']}/proxy/api/v1/broken") as response:
                iterator=response.aiter_bytes()
                assert await anext(iterator)==b'{"type":"ADDED"}\n'
                env.break_stream.set()
                with pytest.raises(httpx.RemoteProtocolError):
                    await anext(iterator)
    asyncio.run(check())


@pytest.mark.parametrize('error,code',[(asyncio.TimeoutError('fixture'),504),(__import__('aiohttp').ClientOSError(111,'fixture'),503)])
def test_raw_proxy_initial_transport_errors_have_http_status(error,code):
    from gpustack.api.responses import StreamingResponseWithStatusCode
    async def check():
        async def upstream():
            raise error
            yield b'',{},200
        events=[]
        async def send(event):events.append(event)
        await StreamingResponseWithStatusCode(upstream()).stream_response(send)
        assert events[0]['status']==code
        assert json.loads(events[-1]['body'])['error']['code']==code
    asyncio.run(check())
