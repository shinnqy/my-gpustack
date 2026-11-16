import asyncio
import base64
import json
import logging
from multiprocessing import Process
from pathlib import Path
import sys
from types import SimpleNamespace

import aiohttp
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
import yaml

from gpustack.config.config import Config
from gpustack.gpu_instances import gateway, gateway_client
from gpustack.gpu_instances.gateway import OperatorSubscriptionReconciler
from gpustack.k8s.manifest_template import TemplateConfig
from gpustack.schemas.clusters import Cluster, ClusterProvider, K8sOptions
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server import db
from gpustack.server.bus import Event, EventType, event_bus
from gpustack.server.init_db import create_db_and_tables
from gpustack.server.server import Server, _ExternalSubprocess
from gpustack_runtime.detector import ManufacturerEnum

from test_watch import project_context, eventually
from test_resource_fit import worker
from test_vllm_launch import launch_server


def documents(text):
    return [item for item in yaml.safe_load_all(text) if item]


@pytest.mark.parametrize('runtimes', [[],['nvidia'],['ascend'],['nvidia','ascend','nvidia'],
    ['amd','ascend','cambricon','hygon','iluvatar','metax','mthreads','nvidia','thead']])
def test_manifests_render_valid_vendor_daemonsets_and_defaults(tmp_path,monkeypatch,runtimes):
    async def check():
        async with project_context(tmp_path,monkeypatch) as env:
            response=await env.client.post('/v2/clusters',json={'name':'kube','provider':'Kubernetes','server_url':'http://server.example:30080'})
            assert response.status_code==200,response.text
            cluster=response.json()
            mounts=cluster['k8s_options']['volumeMounts']
            assert mounts[0]['name']=='gpustack-data-dir' and mounts[0]['mountPath']=='/var/lib/gpustack'
            response=await env.client.get(f"/v2/clusters/{cluster['id']}/manifests",params=[('runtime',value) for value in runtimes])
            assert response.status_code==200,response.text
            items=documents(response.text)
            sets=[item for item in items if item['kind']=='DaemonSet']
            assert {item['metadata']['name'] for item in sets}=={'gpustack-worker'}|{'gpustack-worker-'+value for value in runtimes}
            service=next(item for item in items if item['kind']=='Service' and item['metadata']['name']=='gpustack-worker')
            assert [item['port'] for item in service['spec']['ports']]==[10150,10151]
            for item in sets:
                spec=item['spec']['template']['spec']
                container=spec['containers'][0]
                values={item['name']:item.get('value') for item in container['env']}
                assert values['GPUSTACK_RUNTIME_DEPLOY']=='Kubernetes'
                assert values['GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT']=='true'
                assert spec['enableServiceLinks'] is False
                assert any(volume['name']=='gpustack-data-dir' for volume in spec['volumes'])
                assert any(mount['mountPath']=='/var/lib/gpustack' for mount in container['volumeMounts'])
                assert item['metadata']['namespace']=='gpustack-system'
                if item['metadata']['name'].endswith('-nvidia'):
                    assert spec['runtimeClassName']=='nvidia'
                    assert spec['nodeSelector']['feature.node.kubernetes.io/pci-10de.present']=='true'
                if item['metadata']['name'].endswith('-ascend'):
                    assert spec['nodeSelector']['feature.node.kubernetes.io/pci-19e5.present']=='true'
                    assert any(mount['mountPath']=='/usr/local/Ascend/driver' for mount in container['volumeMounts'])
            secret=next(item for item in items if item['kind']=='Secret' and item['metadata']['name']=='registration-token')
            token=(await env.client.get(f"/v2/clusters/{cluster['id']}/registration-token")).json()['token']
            assert base64.b64decode(secret['data']['GPUSTACK_TOKEN']).decode()==token
    asyncio.run(check())


def test_manifest_overrides_secrets_tristate_and_render_does_not_persist_defaults(tmp_path,monkeypatch):
    async def check():
        async with project_context(tmp_path,monkeypatch) as env:
            env.cfg.namespace='custom-ns'
            env.cfg.operator_image='quay.io/example/operator:fixture'
            payload={'name':'kube','provider':'Kubernetes','system_default_container_registry':'mirror.test',
                'worker_config':{'worker_port':11050,'worker_metrics_port':11051},
                'k8s_options':{'nodeSelector':{'room':'test'},'imageCredentials':[{'registry':'mirror.test','username':'user','password':'p:a\nss'}],
                    'gpuInstanceOptions':{'gpuInstanceTypeDerivedFromNode':False},'operator':{'env':{'EXAMPLE':'true: quoted'}}}}
            response=await env.client.post('/v2/clusters',json=payload)
            assert response.status_code==200,response.text
            cluster=response.json()
            response=await env.client.get(f"/v2/clusters/{cluster['id']}/manifests?runtime=nvidia")
            assert response.status_code==200,response.text
            items=documents(response.text)
            secret=next(item for item in items if item['kind']=='Secret' and item.get('type')=='kubernetes.io/dockerconfigjson')
            credentials=json.loads(base64.b64decode(secret['data']['.dockerconfigjson']))
            assert credentials['auths']['mirror.test']['password']=='p:a\nss'
            configmap=next(item for item in items if item['kind']=='ConfigMap' and item['metadata']['name']=='gpustack-operator-worker-deployment')
            deployment=next(document for content in configmap['data'].values() if 'kind: Deployment' in content
                for document in documents(content) if document['kind']=='Deployment')
            assert deployment['spec']['template']['spec']['containers'][0]['image']=='quay.io/example/operator:fixture'
            values={item['name']:item.get('value') for item in deployment['spec']['template']['spec']['containers'][0]['env']}
            assert values['GPUSTACK_INSTANCE_TYPE_DERIVED_FROM_NODE']=='false'
            assert values['EXAMPLE']=='true: quoted'
            for item in items:
                if item['kind']=='DaemonSet':
                    spec=item['spec']['template']['spec']
                    assert spec['nodeSelector']['room']=='test'
                    assert spec['containers'][0]['ports'][0]['containerPort']==11050
            loaded=(await env.client.get(f"/v2/clusters/{cluster['id']}")).json()
            assert 'namespace' not in loaded['k8s_options'] and 'operatorImage' not in loaded['k8s_options']
            assert loaded['k8s_options']['gpuInstanceOptions']['gpuInstanceTypeDerivedFromNode'] is False
            assert (await env.client.put(f"/v2/clusters/{cluster['id']}",json={'name':'renamed'})).status_code==200
            assert (await env.client.get(f"/v2/clusters/{cluster['id']}")).json()['k8s_options']['gpuInstanceOptions']
    asyncio.run(check())


def test_manifest_provider_validation_and_mount_constraints(tmp_path,monkeypatch):
    async def check():
        async with project_context(tmp_path,monkeypatch) as env:
            docker=(await env.client.post('/v2/clusters',json={'name':'docker'})).json()
            assert (await env.client.get(f"/v2/clusters/{docker['id']}/manifests")).status_code==422
            assert (await env.client.get('/v2/clusters/999/manifests')).status_code==404
            payload={'name':'kube','provider':'Kubernetes','k8s_options':{'volumeMounts':[{
                'name':'data','mountPath':'/ignored','readOnly':True,'volumeSource':{'hostPath':{'path':'/disk/cache'}}}]}}
            response=await env.client.post('/v2/clusters',json=payload)
            assert response.status_code==200,response.text
            mount=response.json()['k8s_options']['volumeMounts'][0]
            assert mount['mountPath']=='/var/lib/gpustack' and mount['readOnly'] is False
            assert mount['volumeSource']['hostPath']['path']=='/disk/cache'
            with_config={**payload,'name':'with-config','k8s_options':{'volumeMounts':[
                *payload['k8s_options']['volumeMounts'],
                {'name':'extra-config','mountPath':'/config','readOnly':True,'volumeSource':{'configMap':{'name':'fixture'}}},
            ]}}
            assert (await env.client.post('/v2/clusters',json=with_config)).status_code==200
            payload['name']='invalid'
            payload['k8s_options']['volumeMounts'][0]['volumeSource']={'persistentVolumeClaim':{'claimName':'pvc'}}
            assert (await env.client.post('/v2/clusters',json=payload)).status_code==422
            assert (await env.client.get(f"/v2/clusters/{response.json()['id']}/manifests?runtime=invalid")).status_code==422
    asyncio.run(check())


def test_old_cluster_schema_upgrade_retains_existing_rows_and_tokens(tmp_path):
    async def check():
        engine=create_async_engine('sqlite+aiosqlite:///'+str(tmp_path/'old.sqlite'))
        try:
            async with engine.begin() as connection:
                await connection.exec_driver_sql('CREATE TABLE clusters (id INTEGER PRIMARY KEY, name VARCHAR, registration_token VARCHAR)')
                await connection.exec_driver_sql("INSERT INTO clusters VALUES (1, 'existing', 'kept-token')")
            await create_db_and_tables(engine)
            await create_db_and_tables(engine)
            async with engine.connect() as connection:
                row=(await connection.exec_driver_sql('SELECT name, registration_token, k8s_options FROM clusters')).one()
                assert tuple(row)==('existing','kept-token',None)
        finally:await engine.dispose()
    asyncio.run(check())


async def seed_eligible(env):
    response=await env.client.post('/v2/clusters',json={'name':'operator','provider':'Kubernetes','k8s_options':{'gpuInstanceOptions':{}}})
    assert response.status_code==200,response.text
    async with db.async_session() as session:
        return await Cluster.one_by_id(session,response.json()['id'])


def test_operator_transitions_retry_token_rotation_and_id_only_delete(tmp_path,monkeypatch):
    async def check():
        async with project_context(tmp_path,monkeypatch) as env:
            cluster=await seed_eligible(env)
            calls=[];fail_delete=[True]
            async def subscribe(cluster,token,**kwargs):calls.append(('subscribe',cluster,token,kwargs))
            async def unsubscribe(cluster):
                calls.append(('unsubscribe',cluster))
                if fail_delete[0]:
                    fail_delete[0]=False
                    raise aiohttp.ClientConnectionError('fixture')
            monkeypatch.setattr(gateway_client,'subscribe_worker',subscribe)
            monkeypatch.setattr(gateway_client,'unsubscribe_worker',unsubscribe)
            reconciler=OperatorSubscriptionReconciler()
            await reconciler._reconcile_cluster(Event(type=EventType.CREATED,data=cluster))
            assert not calls
            async with db.async_session() as session:
                node=await Worker.create(session,worker(1,cluster_id=cluster.id))
            await reconciler._reconcile_worker(Event(type=EventType.CREATED,data=node))
            await reconciler._reconcile_cluster(Event(type=EventType.UPDATED,data=cluster))
            assert len(calls)==1 and reconciler._subscribed=={cluster.id}
            cluster.registration_token='rotated'
            await reconciler._reconcile_cluster(Event(type=EventType.UPDATED,data=cluster))
            assert calls[-1][3]['force'] is True and calls[-1][2]=='rotated'
            async with db.async_session() as session:
                stored=await Worker.one_by_id(session,node.id)
                await stored.delete(session)
            # An id-only worker deletion is repaired on the heartbeat DB scan.
            await reconciler._reconcile_worker(Event(type=EventType.DELETED,data={'id':node.id}))
            await reconciler._retry_stale_subscriptions()
            assert cluster.id in reconciler._subscribed
            await reconciler._retry_stale_subscriptions()
            assert not reconciler._subscribed
    asyncio.run(check())


def test_operator_retry_and_watcher_shutdown_do_not_leak_subscriptions(tmp_path,monkeypatch):
    async def check():
        async with project_context(tmp_path,monkeypatch) as env:
            cluster=await seed_eligible(env)
            async with db.async_session() as session:await Worker.create(session,worker(1,cluster_id=cluster.id))
            attempts=[]
            async def subscribe(*args,**kwargs):
                attempts.append(args)
                if len(attempts)==1:raise aiohttp.ClientConnectionError('fixture')
            monkeypatch.setattr(gateway_client,'subscribe_worker',subscribe)
            reconciler=OperatorSubscriptionReconciler()
            with pytest.raises(aiohttp.ClientConnectionError):
                await reconciler._reconcile_cluster(Event(type=EventType.CREATED,data=cluster))
            await reconciler._reconcile_cluster(Event(type=EventType.HEARTBEAT,data=None))
            assert len(attempts)==2 and cluster.id in reconciler._subscribed
            task=asyncio.create_task(OperatorSubscriptionReconciler().start())
            await eventually(lambda:any(s.source==gateway._SOURCE for s in event_bus.subscribers.get('worker',[])))
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            assert not any(s.source==gateway._SOURCE for subscribers in event_bus.subscribers.values() for s in subscribers)
    asyncio.run(check())


def test_external_process_logs_exit_cleanup_and_no_binary_path(tmp_path,monkeypatch,caplog):
    callbacks=[]
    process=_ExternalSubprocess([sys.executable,'-u','-c','import time; print("stdout-fixture"); import sys; print("stderr-fixture",file=sys.stderr); time.sleep(30)'],
        name='fixture',on_exit=lambda:callbacks.append(True))
    with caplog.at_level(logging.INFO):
        process.start()
        try:
            assert process.is_alive()
        finally:process.stop()
        process.stop()
    assert process.exitcode is not None and callbacks==[True]
    server=Server(Config(data_dir=str(tmp_path)),Process())
    monkeypatch.setattr('gpustack.server.server.shutil.which',lambda name:None)
    server._enqueue_operator_process()
    server._start_gpustack_operator_subscription()
    assert not server.all_processes and not server._async_tasks


def test_operator_socket_is_private_short_and_cleaned_on_failed_start(tmp_path,monkeypatch):
    server=Server(Config(data_dir=str(tmp_path)),Process())
    monkeypatch.setattr('gpustack.server.server.shutil.which',lambda name:'/fixture/operator')
    server._enqueue_operator_process()
    process=server.all_processes[0]
    socket_path=Path(process._args[process._args.index('--bind-unix-path')+1])
    assert len(str(socket_path).encode())<104 and socket_path.parent.stat().st_mode & 0o777==0o700
    monkeypatch.setattr('gpustack.server.server.subprocess.Popen',lambda *args,**kwargs:(_ for _ in ()).throw(OSError('fixture')))
    with pytest.raises(OSError):process.start()
    assert not socket_path.parent.exists()
    gateway_client.set_gateway_unix_path(None)


def test_runtime_kubernetes_conversion_keeps_selected_devices_and_mirrored_storage(launch_server,monkeypatch):
    from gpustack_runtime.deployer import kuberentes as runtime_kube
    from gpustack_runtime.deployer.__types__ import DevicesMaterial
    from gpustack_runtime import envs as runtime_envs
    from gpustack.worker.backends import vllm
    import kubernetes.client as k8s
    context=launch_server()
    node_name='fixture-node'
    self_pod=k8s.V1Pod(metadata=k8s.V1ObjectMeta(name='worker',namespace='gpustack-system'),spec=k8s.V1PodSpec(
        containers=[k8s.V1Container(name='default',env=[k8s.V1EnvVar(name='CACHE_HINT',value='fixture')],
            volume_mounts=[k8s.V1VolumeMount(name='data',mount_path='/var/lib/gpustack')])],
        volumes=[k8s.V1Volume(name='data',host_path=k8s.V1HostPathVolumeSource(path='/var/lib/gpustack'))],
        image_pull_secrets=[k8s.V1LocalObjectReference(name='private-registry')]))
    captured=[]
    def create_pod(namespace,body):
        captured.append(body)
        return body
    core=SimpleNamespace(read_node=lambda **kwargs:k8s.V1Node(metadata=k8s.V1ObjectMeta(name=node_name),
        status=k8s.V1NodeStatus(allocatable={'nvidia.com/gpu.shared':'4'})),
        list_node=lambda **kwargs:k8s.V1NodeList(items=[k8s.V1Node(metadata=k8s.V1ObjectMeta(name=node_name),status=k8s.V1NodeStatus(allocatable={'nvidia.com/gpu.shared':'4'}))]),
        read_namespaced_pod=lambda **kwargs:None,create_namespaced_pod=create_pod)
    monkeypatch.setattr(k8s,'CoreV1Api',lambda *args,**kwargs:core)
    monkeypatch.setattr(k8s,'NodeV1Api',lambda *args,**kwargs:SimpleNamespace(read_runtime_class=lambda **kwargs:None))
    deployer=object.__new__(runtime_kube.KubernetesDeployer)
    deployer._client=None;deployer._node_name=node_name;deployer._image_pull_secret=None;deployer._mutate_create_pod=None
    runtime_name=runtime_envs.GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES['nvidia.com/devices']
    deployer._materials={runtime_name:DevicesMaterial(manufacturer=ManufacturerEnum.NVIDIA,runtime_env=runtime_name,
        cdi='nvidia.com/gpu',runtime_values={'0':'GPU-0','1':'GPU-1'})}
    monkeypatch.setattr(deployer,'_find_self_pod',lambda:self_pod)
    monkeypatch.setattr(runtime_envs,'GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT',True)
    monkeypatch.setattr(runtime_envs,'GPUSTACK_RUNTIME_DEPLOY','Kubernetes')
    monkeypatch.setattr(runtime_envs,'GPUSTACK_RUNTIME_KUBERNETES_NAMESPACE','gpustack-system')
    def convert(plan):
        deployer._prepare_mirrored_deployment()
        workload=runtime_kube.KubernetesWorkloadPlan(**plan.__dict__)
        workload.validate_and_default()
        return deployer._create_pod(workload,{})
    monkeypatch.setattr(vllm,'create_workload',convert)
    context.server.start()
    pod=captured[0];container=pod.spec.containers[0]
    assert pod.spec.node_name==node_name and pod.spec.host_network is True
    assert next(volume for volume in pod.spec.volumes if volume.name=='dshm').empty_dir.size_limit==str(10<<30)
    assert container.resources.requests['nvidia.com/gpu.shared']=='1'
    assert pod.metadata.annotations['device.gpustack.ai/accelerator.preferred-index']=='1'
    assert any(mount.mount_path=='/var/lib/gpustack' for mount in container.volume_mounts)
    assert any(item.name=='CACHE_HINT' and item.value=='fixture' for item in container.env)
    assert pod.spec.image_pull_secrets[0].name=='private-registry'


def test_operator_failure_is_monitored_and_socket_cleanup_is_once(tmp_path):
    async def check():
        cleanup=[]
        handle=_ExternalSubprocess([sys.executable,'-c','raise SystemExit(7)'],name='operator-fixture',on_exit=lambda:cleanup.append(True))
        server=Server(Config(data_dir=str(tmp_path)),Process())
        server._before_healthy_sub_processes.append(handle)
        handle.start()
        await asyncio.to_thread(handle._popen.wait)
        try:
            with pytest.raises(RuntimeError,match='exit code 7'):
                await server._monitor_sub_processes()
        finally:handle.stop()
        assert cleanup==[True]
    asyncio.run(check())
