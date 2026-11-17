import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
import yaml

from gpustack.cloud_providers.abstract import ProviderClientBase,CloudInstance,InstanceState,InstanceProvisioningFailed
from gpustack.cloud_providers.common import generate_ssh_key_pair,creation_idempotency_key
from gpustack.cloud_providers.user_data import UserDataTemplate
from gpustack.schemas.clusters import Cluster,CloudCredential,WorkerPool,Credential
from gpustack.schemas.workers import Worker,WorkerStateEnum
from gpustack.server import db,controllers
from gpustack.server.bus import Event,EventType,event_bus
from gpustack.server.controllers import WorkerPoolController,WorkerProvisioningController,new_workers_from_pool
from gpustack.server.init_db import create_db_and_tables
from gpustack.server.services import WorkerService
from gpustack_runtime.detector import ManufacturerEnum

from test_watch import project_context,eventually
from test_worker_registration import worker_payload


class FakeCloud(ProviderClientBase):
    def __init__(self):
        self.instances={};self.keys={};self.calls=[];self.created=[];self.fail_delete=False;self.terminal=False;self.closed=0
    async def create_instance(self,request):
        self.calls.append(('create',request.name))
        existing=next((vm for vm in self.instances.values() if vm.idempotency_key==request.idempotency_key),None)
        if existing:return existing.external_id
        identifier=str(len(self.created)+1)
        vm=CloudInstance(name=request.name,image=request.image,type=request.type,region=request.region,
            external_id=identifier,status=InstanceState.RUNNING,ip_address='198.51.100.1',
            volume_ids=[],idempotency_key=request.idempotency_key,ssh_endpoint=('ssh.example',2222),ssh_user='root')
        self.instances[identifier]=vm;self.created.append(request)
        return identifier
    async def get_instance(self,identifier):
        if self.terminal:raise InstanceProvisioningFailed('fixture sold out')
        return self.instances.get(identifier)
    async def wait_for_started(self,identifier,**kwargs):return await self.get_instance(identifier)
    async def wait_for_public_ip(self,identifier,**kwargs):return await self.get_instance(identifier)
    async def create_ssh_key(self,name,public_key):
        identifier='key-'+name;self.keys[identifier]=public_key;self.calls.append(('ssh-create',name));return identifier
    async def delete_ssh_key(self,identifier):
        self.calls.append(('ssh-delete',identifier));self.keys.pop(identifier,None)
    async def create_volumes_and_attach(self,worker_id,identifier,region,*volumes):
        identifiers=[f'volume-{identifier}-{i}' for i,_ in enumerate(volumes)]
        self.instances[identifier].volume_ids=identifiers;self.calls.append(('volumes',identifier))
        return identifiers
    async def delete_instance(self,identifier):
        self.calls.append(('delete',identifier))
        if self.fail_delete:raise RuntimeError('fixture delete rejected')
        self.instances.pop(identifier,None)
    async def aclose(self):self.closed+=1


@asynccontextmanager
async def cloud_context(tmp_path,monkeypatch,replicas=2,batch=1,volumes=None):
    async with project_context(tmp_path,monkeypatch,database_file=True) as env:
        fake=FakeCloud()
        monkeypatch.setattr(controllers,'get_client_from_provider',lambda *args,**kwargs:fake)
        async def reachable(*args,**kwargs):return
        monkeypatch.setattr(WorkerProvisioningController,'check_server_external_url',reachable)
        credential=(await env.client.post('/v2/cloud-credentials',json={'name':'fixture','provider':'DigitalOcean','secret':'do-fixture-secret'})).json()
        assert 'secret' not in credential and 'do-fixture-secret' not in str(credential)
        pool={'name':'gpu','instance_type':'gpu-fixture','os_image':'gpu-image','image_name':'registry.test/worker:chapter16',
              'replicas':replicas,'batch_size':batch,'labels':{'purpose':'fixture'}}
        if volumes:pool['cloud_options']={'volumes':volumes}
        response=await env.client.post('/v2/clusters',json={'name':'cloud','provider':'DigitalOcean','credential_id':credential['id'],
            'region':'tor1','server_url':'http://server.example','worker_pools':[pool]})
        assert response.status_code==200,response.text
        cluster=response.json()
        pools=(await env.client.get('/v2/worker-pools')).json()['items'];assert len(pools)==1
        manager=WorkerPoolController();provisioner=WorkerProvisioningController(env.cfg)
        yield SimpleNamespace(**vars(env),cloud=fake,cluster=cluster,pool=pools[0],credential=credential,
                              manager=manager,provisioner=provisioner)


async def reconcile_pool(env):
    await env.manager._reconcile(Event(type=EventType.UPDATED,data={'id':env.pool['id']}))


async def progress(env,identifier,target='initializing'):
    for _ in range(14):
        row=(await env.client.get(f'/v2/workers/{identifier}')).json()
        if row['state']==target:return row
        await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':identifier}))
    pytest.fail(f'Worker {identifier} never reached {target}: {row}')


def test_cloud_crud_embedded_pool_counts_and_reference_guards(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch) as env:
            assert (await env.client.delete(f"/v2/cloud-credentials/{env.credential['id']}")).status_code==409
            assert (await env.client.delete(f"/v2/clusters/{env.cluster['id']}")).status_code==409
            await reconcile_pool(env)
            pool=(await env.client.get(f"/v2/worker-pools/{env.pool['id']}")).json()
            assert pool['workers']==1 and pool['ready_workers']==0
            assert (await env.client.delete(f"/v2/worker-pools/{env.pool['id']}")).status_code==403
            docker=(await env.client.post('/v2/clusters',json={'name':'docker'})).json()
            response=await env.client.post(f"/v2/clusters/{docker['id']}/worker-pools",json={k:v for k,v in env.pool.items() if k not in {'id','cluster_id','created_at','updated_at'}})
            assert response.status_code==422
            public=await env.client.get('/v2/cloud-credentials')
            assert 'do-fixture-secret' not in public.text
            assert (await env.client.put(f"/v2/cloud-credentials/{env.credential['id']}",json={'name':'renamed','provider':'DigitalOcean'})).status_code==200
            async with db.async_session() as session:
                assert (await CloudCredential.one_by_id(session,env.credential['id'])).secret=='do-fixture-secret'
    asyncio.run(check())


def test_batch_cap_counts_pending_rows_and_releases_at_initializing(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch,replicas=4,batch=2) as env:
            await reconcile_pool(env)
            rows=(await env.client.get('/v2/workers')).json()['items'];assert len(rows)==2
            await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':rows[0]['id']}))
            await reconcile_pool(env)
            assert len((await env.client.get('/v2/workers')).json()['items'])==2
            await progress(env,rows[0]['id'])
            await reconcile_pool(env)
            assert len((await env.client.get('/v2/workers')).json()['items'])==3
    asyncio.run(check())


def test_provisioning_persists_keys_volumes_endpoint_and_registration_claim(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch,volumes=[{'size_gb':10,'format':'ext4'}]) as env:
            await reconcile_pool(env)
            row=(await env.client.get('/v2/workers')).json()['items'][0]
            ready=await progress(env,row['id'])
            assert ready['external_id']=='1' and ready['provider']=='DigitalOcean'
            assert ready['provider_config']['volume_ids']==['volume-1-0']
            assert ready['provider_config']['ssh_endpoint']=={'host':'ssh.example','port':2222,'user':'root'}
            assert ready['provider_config']['provisioning_complete'] is True
            cloud_init=yaml.safe_load(env.cloud.created[0].user_data)
            config=yaml.safe_load(next(file['content'] for file in cloud_init['write_files'] if file['path'].endswith('config.yaml')))
            assert config['worker_name']==row['name'] and config['server_url']=='http://server.example'
            assert 'registry.test/worker:chapter16' in env.cloud.created[0].user_data
            registration=(await env.client.get(f"/v2/clusters/{env.cluster['id']}/registration-token")).json()
            response=await env.client.post('/v2/workers',headers={'Authorization':'Bearer '+registration['token']},
                json=worker_payload(row['name'],worker_uuid=''))
            assert response.status_code==200,response.text
            joined=response.json()
            assert joined['id']==row['id'] and joined['state']=='ready'
            assert joined['external_id']=='1' and joined['worker_pool_id']==env.pool['id']
            assert joined['advertise_address']=='198.51.100.1'
            assert len((await env.client.get('/v2/workers')).json()['items'])==1
            assert (await env.client.get(f"/v2/worker-pools/{env.pool['id']}")).json()['ready_workers']==1
            assert env.cloud.closed>0
    asyncio.run(check())


def test_early_registration_does_not_expose_worker_before_cloud_finalization(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch,replicas=1) as env:
            await reconcile_pool(env)
            row=(await env.client.get('/v2/workers')).json()['items'][0]
            for _ in range(3):await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':row['id']}))
            row=(await env.client.get(f"/v2/workers/{row['id']}")).json();assert row['external_id']=='1'
            token=(await env.client.get(f"/v2/clusters/{env.cluster['id']}/registration-token")).json()['token']
            response=await env.client.post('/v2/workers',headers={'Authorization':'Bearer '+token},json=worker_payload(row['name']))
            assert response.status_code==200,response.text
            assert response.json()['state']=='provisioning'
            complete=await progress(env,row['id'],target='ready')
            assert complete['provider_config']['provisioning_complete'] is True
    asyncio.run(check())


def test_delete_failure_keeps_resource_handles_and_can_be_retried(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch,replicas=1) as env:
            await reconcile_pool(env)
            row=(await env.client.get('/v2/workers')).json()['items'][0]
            row=await progress(env,row['id'])
            env.cloud.fail_delete=True
            assert (await env.client.delete(f"/v2/workers/{row['id']}")).status_code==200
            await env.provisioner._reconcile(Event(type=EventType.DELETED,data={'id':row['id']}))
            failed=(await env.client.get(f"/v2/workers/{row['id']}")).json()
            assert failed['state']=='error' and failed['external_id']=='1' and env.cloud.keys
            async with db.async_session() as session:assert await Credential.one_by_id(session,failed['ssh_key_id'])
            env.cloud.fail_delete=False
            assert (await env.client.delete(f"/v2/workers/{row['id']}")).status_code==200
            await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':row['id']}))
            assert (await env.client.get(f"/v2/workers/{row['id']}")).status_code==404
            assert not env.cloud.instances and not env.cloud.keys
            async with db.async_session() as session:assert not await Credential.all(session)
            await reconcile_pool(env)
            assert len((await env.client.get('/v2/workers')).json()['items'])==1
    asyncio.run(check())


def test_failed_creation_with_only_ssh_key_can_be_cleaned(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch,replicas=1) as env:
            await reconcile_pool(env);row=(await env.client.get('/v2/workers')).json()['items'][0]
            for _ in range(2):await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':row['id']}))
            assert env.cloud.keys and not env.cloud.instances
            await env.client.delete(f"/v2/workers/{row['id']}")
            await env.provisioner._reconcile(Event(type=EventType.DELETED,data={'id':row['id']}))
            assert not env.cloud.keys and not (await env.client.get('/v2/workers')).json()['items']
    asyncio.run(check())


def test_scale_down_removes_workers_without_recreating_them(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch,replicas=2,batch=2) as env:
            await reconcile_pool(env)
            for row in (await env.client.get('/v2/workers')).json()['items']:await progress(env,row['id'])
            assert (await env.client.put(f"/v2/worker-pools/{env.pool['id']}",json={'name':env.pool['name'],'replicas':0})).status_code==200
            await reconcile_pool(env)
            rows=(await env.client.get('/v2/workers')).json()['items'];assert all(row['state']=='deleting' for row in rows)
            for row in rows:await env.provisioner._reconcile(Event(type=EventType.DELETED,data={'id':row['id']}))
            await reconcile_pool(env)
            assert not (await env.client.get('/v2/workers')).json()['items']
            assert not env.cloud.instances and not env.cloud.keys
            assert (await env.client.delete(f"/v2/worker-pools/{env.pool['id']}")).status_code==200
            assert (await env.client.delete(f"/v2/clusters/{env.cluster['id']}")).status_code==200
            assert (await env.client.delete(f"/v2/cloud-credentials/{env.credential['id']}")).status_code==200
    asyncio.run(check())


def test_health_check_failure_prevents_any_cloud_resource_creation(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch,replicas=1) as env:
            await reconcile_pool(env);row=(await env.client.get('/v2/workers')).json()['items'][0]
            async def unreachable(*args,**kwargs):raise ValueError('fixture healthz unavailable')
            monkeypatch.setattr(env.provisioner,'check_server_external_url',unreachable)
            await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':row['id']}))
            failed=(await env.client.get(f"/v2/workers/{row['id']}")).json()
            assert failed['state']=='error' and 'healthz' in failed['state_message'] and not env.cloud.calls
    asyncio.run(check())


@pytest.mark.parametrize('driver',[None,ManufacturerEnum.NVIDIA])
def test_cloud_init_quotes_secrets_and_is_repeatable(driver):
    template=UserDataTemplate('http://server:30080','token"\nvalue','registry.test/worker:latest','pool-1-node',
                              secret_configs={'huggingface_token':'hf"\nvalue: true'})
    template.install_driver=driver;template.setup_driver=driver
    first=template.format();assert template.format()==first
    data=yaml.safe_load(first)
    config=yaml.safe_load(next(file['content'] for file in data['write_files'] if file['path'].endswith('config.yaml')))
    assert config['token']=='token"\nvalue' and config['huggingface_token']=='hf"\nvalue: true'
    assert data['packages'].count('docker.io')==1
    assert bool(data.get('power_state')) is (driver is not None)


def test_generated_ssh_keys_match_and_idempotency_uses_persisted_identity():
    private,public=generate_ssh_key_pair()
    key=Ed25519PrivateKey.from_private_bytes(base64.b64decode(private))
    assert key.public_key().public_bytes(serialization.Encoding.OpenSSH,serialization.PublicFormat.OpenSSH).decode()==public
    from test_resource_fit import worker
    node=worker(1)
    node.updated_at=datetime(2026,1,1,tzinfo=timezone.utc)
    original=creation_idempotency_key(node)
    node.updated_at=node.updated_at.replace(tzinfo=None)
    assert creation_idempotency_key(node)==original
    node.updated_at+=timedelta(seconds=1)
    assert creation_idempotency_key(node)!=original


def test_old_database_upgrades_pool_fields_without_losing_workers(tmp_path):
    async def check():
        engine=create_async_engine('sqlite+aiosqlite:///'+str(tmp_path/'legacy.sqlite'))
        try:
            async with engine.begin() as connection:
                await connection.exec_driver_sql('CREATE TABLE clusters (id INTEGER PRIMARY KEY, name TEXT, registration_token TEXT, provider TEXT)')
                await connection.exec_driver_sql("INSERT INTO clusters VALUES (1, 'kept', 'kept-token', 'Docker')")
                await connection.exec_driver_sql('CREATE TABLE workers (id INTEGER PRIMARY KEY, name TEXT, cluster_id INTEGER)')
                await connection.exec_driver_sql("INSERT INTO workers VALUES (1, 'kept-worker', 1)")
            await create_db_and_tables(engine)
            await create_db_and_tables(engine)
            async with engine.connect() as connection:
                row=(await connection.exec_driver_sql('SELECT name, provider, worker_pool_id, external_id, ssh_key_id, provider_config FROM workers')).one()
                assert tuple(row)==('kept-worker','Docker',None,None,None,None)
                assert (await connection.exec_driver_sql('SELECT registration_token, credential_id FROM clusters')).one()==('kept-token',None)
        finally:await engine.dispose()
    asyncio.run(check())


def test_replayed_create_keeps_key_for_a_provider_that_supports_idempotence(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch,replicas=1) as env:
            await reconcile_pool(env);row=(await env.client.get('/v2/workers')).json()['items'][0]
            for _ in range(2):await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':row['id']}))
            from sqlalchemy.orm import selectinload
            async with db.async_session() as session:
                node=await Worker.one_by_id(session,row['id'],options=[selectinload(Worker.cluster),selectinload(Worker.worker_pool)])
                await env.provisioner._create_instances(session,env.cloud,node,env.cfg)
                await session.rollback()  # 云端已接受，Server 尚未保存 external_id 即退出。
            assert len(env.cloud.created)==1
            await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':row['id']}))
            saved=(await env.client.get(f"/v2/workers/{row['id']}")).json()
            assert saved['external_id']=='1' and len(env.cloud.created)==1
    asyncio.run(check())


def test_terminal_provider_failure_is_not_retried_and_rejects_bad_pool_inputs(tmp_path,monkeypatch):
    async def check():
        async with cloud_context(tmp_path,monkeypatch,replicas=1) as env:
            await reconcile_pool(env);row=(await env.client.get('/v2/workers')).json()['items'][0]
            for _ in range(3):await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':row['id']}))
            env.cloud.terminal=True
            await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':row['id']}))
            failed=(await env.client.get(f"/v2/workers/{row['id']}")).json()
            assert failed['state']=='error' and 'sold out' in failed['state_message']
            calls=len(env.cloud.calls)
            await env.provisioner._reconcile(Event(type=EventType.UPDATED,data={'id':row['id']}))
            assert len(env.cloud.calls)==calls
            response=await env.client.put(f"/v2/worker-pools/{env.pool['id']}",json={'name':'gpu','replicas':-1})
            assert response.status_code==422
            response=await env.client.post(f"/v2/clusters/{env.cluster['id']}/worker-pools",json={
                'name':'bad','instance_type':'gpu','os_image':'image','image_name':'image',
                'cloud_options':{'volumes':[{'size_gb':0,'format':'ext4'}]}})
            assert response.status_code==422
    asyncio.run(check())
