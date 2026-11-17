import asyncio
import json
from types import SimpleNamespace
from urllib.parse import urlparse

from azure.core.pipeline.transport import AsyncHttpTransport,AsyncHttpResponse
import httpx
import pytest
from pydo.aio import Client
import yaml

from gpustack.cloud_providers import digital_ocean as ocean
from gpustack.cloud_providers import shuihua
from gpustack.cloud_providers.abstract import CloudInstanceCreate,InstanceState,InstanceProvisioningFailed


class Response(AsyncHttpResponse):
    def __init__(self,request,status,payload):
        super().__init__(request,None)
        self.status_code=status;self.headers={'content-type':'application/json'}
        self.content_type='application/json';self.reason='fixture'
        self._body=json.dumps(payload).encode() if payload is not None else b''
    def body(self):return self._body
    @property
    def content(self):return self._body
    def json(self):return json.loads(self._body)
    async def load_body(self):return
    def stream_download(self,pipeline,**kwargs):
        async def chunks():yield self._body
        return chunks()


class Transport(AsyncHttpTransport):
    def __init__(self):self.requests=[];self.closed=False;self.fail_delete=False
    async def open(self):return
    async def close(self):self.closed=True
    async def __aenter__(self):return self
    async def __aexit__(self,*args):await self.close()
    async def send(self,request,**kwargs):
        path=urlparse(request.url).path
        self.requests.append(request)
        body={};status=200
        if path.endswith('/droplets') and request.method=='POST':body={'droplet':{'id':42}};status=202
        elif path.endswith('/droplets/42'):
            body={'droplet':{'id':42,'name':'node','image':{'slug':'gpu-image'},'size_slug':'gpu','region':{'slug':'tor1'},'status':'active',
                'networks':{'v4':[{'type':'public','ip_address':'198.51.100.42'}]},'volume_ids':[]}}
        elif '/images/' in path:body={'image':{'distribution':'Ubuntu','slug':'gpu-h100-image'}}
        elif path.endswith('/account/keys') and request.method=='POST':body={'ssh_key':{'id':7}};status=201
        elif request.method=='DELETE':
            status=503 if self.fail_delete else (202 if 'dangerous' in path else 204);body={'message':'busy'} if self.fail_delete else None
        return Response(request,status,body)


def test_digitalocean_uses_real_async_sdk_and_preserves_lifecycle_payloads(monkeypatch):
    async def check():
        transport=Transport()
        monkeypatch.setattr(ocean,'Client',lambda **kwargs:Client(**kwargs,transport=transport,retry_total=0))
        client=ocean.DigitalOceanClient('fixture-api-token')
        try:
            key=await client.create_ssh_key('node','ssh-ed25519 fixture')
            assert key=='7'
            request=CloudInstanceCreate(name='node',image='gpu-image',type='gpu-h100',region='tor1',
                ssh_key_id=key,user_data='#cloud-config\n',labels={'worker_id':'1'},idempotency_key='stable')
            assert await client.create_instance(request)=='42'
            vm=await client.wait_for_started('42',backoff=0,limit=1)
            assert vm.status==InstanceState.RUNNING and vm.ip_address=='198.51.100.42'
            template=await client.construct_user_data('http://server','token','registry.test/worker:16','gpu-image','node')
            data=yaml.safe_load(template.format())
            assert not data.get('package_upgrade')
            assert any('nvidia-ctk' in command for command in data['runcmd'])
            await client.delete_instance('42');await client.delete_ssh_key(key)
            created=next(record for record in transport.requests if record.method=='POST' and record.url.endswith('/droplets'))
            payload=json.loads(created.body)
            assert payload['ssh_keys']==['7'] and payload['user_data']=='#cloud-config\n'
            assert 'worker_id:1' in payload['tags']
            assert any(record.headers.get('Authorization')=='Bearer fixture-api-token' for record in transport.requests)
        finally:await client.aclose()
        assert transport.closed
    asyncio.run(check())


def test_digitalocean_delete_errors_are_not_accepted_as_success(monkeypatch):
    async def check():
        transport=Transport();transport.fail_delete=True
        monkeypatch.setattr(ocean,'Client',lambda **kwargs:Client(**kwargs,transport=transport,retry_total=0))
        client=ocean.DigitalOceanClient('fixture')
        try:
            with pytest.raises(Exception):await client.delete_instance('42')
        finally:await client.aclose()
    asyncio.run(check())


def test_shuihua_provider_replays_identity_embeds_ssh_and_handles_terminal_states(monkeypatch):
    async def check():
        records=[];state=['creating']
        def request(request):
            records.append(request)
            if request.method=='DELETE':return httpx.Response(200,json={'data':{}})
            return httpx.Response(202 if request.method=='POST' else 200,json={'data':{
                'id':17,'instance_name':'worker','status':state[0],
                'ip_address':'10.0.0.2','port_mappings':[{'public_ip':'198.51.100.2','public_port':2222,'internal_port':22}]}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as http:
            real_client=shuihua.ShuihuaClient
            monkeypatch.setattr(shuihua,'ShuihuaClient',lambda **kwargs:real_client(**kwargs,client=http))
            provider=shuihua.ShuihuaProviderClient('amp_live_fixture',base_url='https://cloud.example')
            template=await provider.construct_user_data('http://server','token','registry.test/worker:16','image-uuid','worker',ssh_public_key='ssh-ed25519 fixture')
            cloud_init=template.format()
            assert yaml.safe_load(cloud_init)['ssh_authorized_keys']==['ssh-ed25519 fixture']
            incoming=CloudInstanceCreate(name='worker',image=' image-uuid ',type='9',region='',
                user_data=cloud_init,idempotency_key='stable-key')
            assert await provider.create_instance(incoming)=='17'
            payload=json.loads(records[-1].content)
            assert payload['idempotency_key']=='stable-key' and payload['image_id']=='image-uuid'
            state[0]='active'
            vm=await provider.wait_for_started('17',backoff=0,limit=1)
            assert vm.ssh_endpoint==('198.51.100.2',2222)
            state[0]='failed'
            with pytest.raises(InstanceProvisioningFailed):await provider.wait_for_started('17',backoff=0,limit=1)
            assert await provider.create_ssh_key('worker','public') is None
            await provider.delete_instance('17')
            assert all(record.headers['Authorization']=='Bearer amp_live_fixture' for record in records)
    asyncio.run(check())
