import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from gpustack.config.config import Config
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import VLLMResourceFitSelector
from gpustack.policies import utils
from gpustack.scheduler.scheduler import Scheduler
from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum, DistributedServers, ModelInstanceSubordinateWorker
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.workers import Worker
from gpustack.server import db
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import ensure_instance_model_file, ModelFileController
from gpustack.worker import serve_manager as serve_module
from gpustack.worker.serve_manager import ServeManager
from gpustack.worker.backends import vllm as vllm_module
from gpustack.utils.attrs import set_attr
from gpustack.utils.command import find_parameter
from gpustack_runtime.deployer import WorkloadStatusStateEnum as RuntimeState

from test_resource_fit import model, worker, instance, config_stub, GiB
from test_health_recovery import manager as health_manager
from test_vllm_launch import launch_server
from test_watch import project_context, seed


@pytest.fixture(autouse=True)
def runtime_fixture(monkeypatch):
    monkeypatch.setattr(serve_module, 'get_workload', lambda name: None)
    monkeypatch.setattr(serve_module, 'delete_workload', lambda name: None)
    monkeypatch.setattr(serve_module, 'logs_workload', lambda **kwargs: iter(()))
    monkeypatch.setattr(serve_module, 'get_meta_from_running_instance', lambda *args: {})
    monkeypatch.setattr(ServeManager, '_start_container_log_persistence', lambda self, mi: None)


def distributed_instance(**changes):
    return instance(state='starting', worker_ip='10.0.0.1', port=40000, ports=[40000, 40011, 40012, 40013, 40001],
        distributed_servers={'mode': 'initialize_later', 'subordinate_workers': [
            {'worker_id': 2, 'worker_name': 'worker-2', 'worker_ip': '10.0.0.2', 'worker_ifname': 'eth0',
             'gpu_type': 'cuda', 'gpu_indexes': [0], 'state': 'pending',
             'computed_resource_claim': {'vram': {0: int(24*GiB*.9)}, 'ram': GiB}}
        ]}, **changes)


@pytest.mark.parametrize('disabled,busy,kind,expected', [
    (False, False, 'cuda', True), (True, False, 'cuda', False),
    (False, True, 'cuda', False), (False, False, 'cann', False),
])
def test_multi_worker_fit_and_exclusions(tmp_path, config_stub, disabled, busy, kind, expected):
    async def check():
        selected_model=model(distributed_inference_across_workers=not disabled, env={'GPUSTACK_MODEL_VRAM_CLAIM': str(40*GiB)})
        workers=[worker(1,sizes=(24,)), worker(2,sizes=(24,),gpu_type=kind)]
        existing=[instance(worker_id=2, vram={0:3*GiB})] if busy else []
        candidates=await VLLMResourceFitSelector(Config(data_dir=str(tmp_path)),selected_model,existing).select_candidates(workers)
        assert bool(candidates) is expected
        if expected:
            selected=candidates[0]
            assert selected.worker.id == 1 and selected.subordinate_workers[0].worker_id == 2
            assert selected.computed_resource_claim.vram == {0:int(24*GiB*.9)}
            assert selected.subordinate_workers[0].computed_resource_claim.vram == {0:int(24*GiB*.9)}
    asyncio.run(check())


def test_explicit_parallelism_requires_exact_number_of_devices(tmp_path, config_stub):
    async def check():
        chosen=model(backend_parameters=['--tensor-parallel-size=1','--pipeline-parallel-size=3'])
        candidates=await VLLMResourceFitSelector(Config(data_dir=str(tmp_path)),chosen,[]).select_candidates([worker(i,sizes=(24,)) for i in range(1,5)])
        assert len(candidates)==1 and len(candidates[0].subordinate_workers)==2
    asyncio.run(check())


def test_follower_claims_are_included_in_ledger_and_worker_view():
    mi=distributed_instance()
    mi.distributed_servers.subordinate_workers[0].state=ModelInstanceStateEnum.RUNNING
    node=worker(2,sizes=(24,))
    assert utils.get_worker_model_instances([mi],node)==[mi]
    allocated=utils.compute_worker_allocated([mi],2,'cuda')
    assert allocated.ram==0 and allocated.vram=={0:int(24*GiB*.9)}
    remaining=utils.get_worker_allocatable_resource([mi],node,'cuda')
    assert remaining.vram[0]==24*GiB-int(24*GiB*.9)


@pytest.mark.parametrize('executor,expected', [('ray',True),('mp',False)])
def test_uneven_manual_workers_validate_executor_topology(tmp_path, config_stub, executor, expected):
    async def check():
        chosen=model(gpu_selector={'gpu_ids':['worker-1:cuda:0','worker-2:cuda:0','worker-2:cuda:1']},
            backend_parameters=['--tensor-parallel-size=1','--pipeline-parallel-size=3','--distributed-executor-backend='+executor],
            env={'GPUSTACK_MODEL_VRAM_CLAIM':str(50*GiB)})
        candidates=await VLLMResourceFitSelector(Config(data_dir=str(tmp_path)),chosen,[]).select_candidates([worker(1,sizes=(24,)),worker(2)])
        assert bool(candidates) is expected
    asyncio.run(check())


def test_scheduler_persists_main_and_subordinate_binding(tmp_path, monkeypatch, config_stub):
    async def check():
        async with project_context(tmp_path,monkeypatch) as env:
            created,raw=await seed(env.client,state='analyzing')
            async with db.async_session() as session:
                for number in (1,2):
                    await Worker.create(session,worker(number,sizes=(24,),cluster_id=created['cluster_id']))
            response=await env.client.put(f"/v2/models/{created['id']}",json={**created,'backend_parameters':['--tensor-parallel-size=1','--pipeline-parallel-size=2']})
            assert response.status_code==200
            await Scheduler(env.cfg)._schedule_one(ModelInstance.model_validate(raw))
            saved=(await env.client.get(f"/v2/model-instances/{raw['id']}")).json()
            assert saved['state']=='scheduled' and saved['worker_id']==1
            assert saved['distributed_servers']['mode']=='initialize_later'
            assert saved['distributed_servers']['subordinate_workers'][0]['worker_id']==2
    asyncio.run(check())


def test_all_workers_model_files_gate_starting_and_track_progress(tmp_path,monkeypatch):
    async def check():
        async with project_context(tmp_path,monkeypatch) as env:
            created,raw=await seed(env.client,state='initializing',worker_id=1)
            raw.update(distributed_servers=distributed_instance().distributed_servers.model_dump(mode='json'))
            assert (await env.client.put(f"/v2/model-instances/{raw['id']}",json=raw)).status_code==200
            async with db.async_session() as session:
                mi=await ModelInstance.one_by_id(session,raw['id'])
                await ensure_instance_model_file(session,mi)
                files=await ModelFile.all(session)
                assert {f.worker_id for f in files}=={1,2}
                for index,file in enumerate(files):
                    await file.update(session,{'state':ModelFileStateEnum.READY,'download_progress':100,'resolved_paths':[f'/worker-{file.worker_id}/model']})
                    await ModelFileController()._reconcile(Event(type=EventType.UPDATED,data=file))
                    state=(await env.client.get(f"/v2/model-instances/{raw['id']}")).json()
                    assert (state['state']=='starting') is (index==1)
                assert state['distributed_servers']['subordinate_workers'][0]['download_progress']==100
    asyncio.run(check())


def test_atomic_worker_patches_preserve_other_nodes_and_reject_old_generation(tmp_path,monkeypatch):
    async def check():
        async with project_context(tmp_path,monkeypatch,database_file=True) as env:
            _,raw=await seed(env.client,state='starting',worker_id=1)
            raw.update(distributed_servers=distributed_instance().distributed_servers.model_dump(mode='json'))
            assert (await env.client.put(f"/v2/model-instances/{raw['id']}",json=raw)).status_code==200
            path=f"/v2/model-instances/{raw['id']}?expected_restart_count=0"
            results=await asyncio.gather(
                env.client.patch(path,json={'state':'running','pid':123}),
                env.client.patch(path,json={'distributed_servers.subordinate_workers.0.state':'running','distributed_servers.subordinate_workers.0.pid':456}),
            )
            assert [r.status_code for r in results]==[200,200]
            saved=(await env.client.get(f"/v2/model-instances/{raw['id']}")).json()
            assert saved['state']=='running' and saved['pid']==123
            assert saved['distributed_servers']['subordinate_workers'][0]['pid']==456
            assert (await env.client.patch(path,json={'restart_count':1})).status_code==200
            assert (await env.client.patch(path,json={'state':'error'})).status_code==409
            assert (await env.client.patch(f"/v2/model-instances/{raw['id']}",json={'distributed_servers.subordinate_workers.99.state':'error'})).status_code==400
    asyncio.run(check())


def test_subordinate_waits_for_main_and_does_not_restart_failed_generation(tmp_path,monkeypatch):
    mi=distributed_instance()
    selected,_,_,_,patches=health_manager(tmp_path,[mi])
    selected._worker_id_getter=lambda:2
    started=[]
    monkeypatch.setattr(selected,'_start_model_instance',lambda mi:started.append(mi.id))
    mi.state=ModelInstanceStateEnum.DOWNLOADING
    selected._dispatch_model_instance_event(Event(type=EventType.UPDATED,data=mi))
    assert not started
    mi.state=ModelInstanceStateEnum.STARTING
    selected._dispatch_model_instance_event(Event(type=EventType.UPDATED,data=mi))
    assert started==[mi.id] and mi.id in selected._model_instance_by_instance_id
    mi.distributed_servers.subordinate_workers[0].state=ModelInstanceStateEnum.ERROR
    selected._dispatch_model_instance_event(Event(type=EventType.UPDATED,data=mi))
    assert started==[mi.id]


def test_subordinate_failure_propagates_and_global_restart_resets_nodes(tmp_path,monkeypatch):
    mi=distributed_instance()
    mi.distributed_servers.subordinate_workers[0].state=ModelInstanceStateEnum.RUNNING
    follower,_,_,_,patches=health_manager(tmp_path/'follower',[mi])
    follower._worker_id_getter=lambda:2
    names=[]
    monkeypatch.setattr(serve_module,'get_workload',lambda name:names.append(name) or SimpleNamespace(state=RuntimeState.FAILED,state_message='fixture failed',exits=[]))
    follower.sync_model_instances_state()
    assert names==[mi.name+'-f0']
    patch=patches[-1][1]
    for key,value in patch.items(): set_attr(mi,key,value)
    assert mi.distributed_servers.subordinate_workers[0].state==ModelInstanceStateEnum.ERROR
    leader,_,_,_,patches=health_manager(tmp_path/'leader',[mi])
    monkeypatch.setattr(serve_module,'get_workload',lambda name:SimpleNamespace(state=RuntimeState.RUNNING))
    leader.sync_model_instances_state()
    assert patches[-1][1]['state']==ModelInstanceStateEnum.ERROR
    assert 'subordinate worker' in patches[-1][1]['state_message']
    mi.state=ModelInstanceStateEnum.ERROR
    leader._restart_error_model_instance(mi)
    restarted=patches[-1][1]
    assert restarted['state']==ModelInstanceStateEnum.SCHEDULED and restarted['restart_count']==1
    assert restarted['distributed_servers'].subordinate_workers[0].state==ModelInstanceStateEnum.PENDING


@pytest.mark.parametrize('executor,parallelism,shape', [
    ('ray',['--tensor-parallel-size=1','--pipeline-parallel-size=2'],'ray'),
    ('mp',['--tensor-parallel-size=1','--pipeline-parallel-size=2'],'mp_only'),
    ('mp',['--data-parallel-size=2'],'dp_only'),
    ('mp',['--tensor-parallel-size=2','--data-parallel-size=2'],'nested'),
])
@pytest.mark.parametrize('follower',[False,True])
def test_distributed_runtime_plans_and_network_environment(launch_server,monkeypatch,executor,parallelism,shape,follower):
    context=launch_server(model_changes={'backend_parameters':parallelism+['--distributed-executor-backend='+executor]})
    payload={**context.instance,'ports':[40001,40012,40013,40014,40002],
        'distributed_servers':{'mode':'initialize_later','subordinate_workers':[{
            'worker_id':2,'worker_ip':'10.0.0.2','worker_name':'follower','worker_ifname':'eth1','gpu_indexes':[1],
            'gpu_type':'cuda','state':'running'}]}}
    context.instance.update(payload)
    if shape=='nested':
        for number in (3,4):
            payload['distributed_servers']['subordinate_workers'].append({
                'worker_id':number,'worker_ip':f'10.0.0.{number}','gpu_indexes':[1],
                'gpu_type':'cuda','state':'running'})
        context.instance.update(payload)
    server=context.server
    server._model_instance=ModelInstance.model_validate(payload)
    if follower:
        server._worker=server._worker.model_copy(update={'id':2,'ip':'10.0.0.2','ifname':'eth1'})
    plans=[]
    monkeypatch.setattr(vllm_module,'create_workload',plans.append)
    server.start()
    plan=plans[0]
    assert plan.name==payload['name']+('-f0' if follower else '')
    run=plan.containers[0]
    env={item.name:item.value for item in run.envs}
    assert env['VLLM_HOST_IP']==server._worker.ip and env['NCCL_SOCKET_IFNAME']=='='+server._worker.ifname
    command=(run.execution.command or [])+(run.execution.args or [])
    if shape=='ray':
        assert len(plan.containers)==(1 if follower else 2)
        if follower:
            assert command[:2]==['ray','start']
            assert '--address=127.0.0.1:41000' in command
        else:
            assert plan.containers[1].name=='ray-head'
            assert any(m.path=='/tmp' and m.volume=='tmp-volume' for m in run.mounts)
            assert find_parameter(command,['distributed-executor-backend'])=='ray'
    else:
        assert len(plan.containers)==1 and command[:2]==['vllm','serve']
        assert ('--headless' in command) is follower
        if shape=='mp_only':
            assert find_parameter(command,['node-rank'])==str(int(follower))
            assert find_parameter(command,['master-port'])=='40013'
            assert 'VLLM_DP_MASTER_PORT' not in env
        elif shape=='dp_only':
            assert find_parameter(command,['data-parallel-start-rank'])==str(int(follower))
            assert env['VLLM_DP_MASTER_PORT']=='40002' and env['VLLM_PORT']=='40014'
        else:
            assert find_parameter(command,['node-rank'])==str(int(follower))
            assert find_parameter(command,['nnodes'])=='4'
            assert find_parameter(command,['data-parallel-rpc-port'])=='40012'
            assert env['VLLM_DP_MASTER_PORT']=='40002' and 'VLLM_PORT' not in env


def test_follower_backend_errors_patch_only_its_own_state(launch_server):
    context=launch_server()
    server=context.server
    server._model_instance=distributed_instance()
    server._worker=server._worker.model_copy(update={'id':2})
    patches=[]
    server._update_model_instance=lambda id,**kwargs:patches.append(kwargs)
    with pytest.raises(RuntimeError,match='fixture'):
        server._handle_error(RuntimeError('fixture failure'))
    assert patches[-1]['distributed_servers.subordinate_workers.0.state']==ModelInstanceStateEnum.ERROR
    assert 'state' not in patches[-1]


def test_numbered_log_cleanup_keeps_current_and_previous(tmp_path):
    manager=ServeManager(lambda:1,lambda:None,Config(data_dir=str(tmp_path)))
    root=Path(manager._serve_log_dir)
    for restart in range(4):
        for name in (f'1.{restart}.log',f'1.container.{restart}.log',f'1.container.ray-head.{restart}.log'):
            (root/name).write_text('fixture')
    manager._cleanup_old_logs(1,3)
    assert len(list(root.glob('*.log')))==6
    manager._cleanup_old_logs(1,0)
    assert not list(root.glob('*.log'))


def test_explicit_cross_node_tp_must_divide_attention_heads(tmp_path,config_stub):
    async def check():
        chosen=model(backend_parameters=['--tensor-parallel-size=3','--distributed-executor-backend=ray'])
        selector=VLLMResourceFitSelector(Config(data_dir=str(tmp_path)),chosen,[])
        assert not await selector.select_candidates([worker(i,sizes=(24,)) for i in range(1,4)])
        assert 'Tensor parallel size 3' in ''.join(selector.get_messages())
    asyncio.run(check())


def test_mp_port_band_is_reserved_between_instances_and_replayed(tmp_path,monkeypatch):
    manager=ServeManager(lambda:1,lambda:None,Config(data_dir=str(tmp_path),service_port_range='43000-43063'))
    chosen=model(backend_parameters=['--distributed-executor-backend=mp'])
    def free(port_range,unavailable_ports,**kwargs):
        start,end=map(int,port_range.split('-'))
        return min(set(range(start,end+1))-unavailable_ports)
    monkeypatch.setattr(serve_module.network,'get_free_port',free)
    monkeypatch.setattr(serve_module.network,'is_port_available',lambda *args:True)
    first=distributed_instance().model_copy(update={'port':None,'ports':[]})
    manager._assign_ports(first,chosen,'vLLM')
    assert len(first.ports)==5 and len(manager._assigned_ports[first.id])==14
    second=first.model_copy(update={'id':2,'port':None,'ports':[]})
    manager._assign_ports(second,chosen,'vLLM')
    assert not manager._assigned_ports[first.id] & manager._assigned_ports[second.id]
    reserved=set(manager._assigned_ports[first.id])
    manager._assigned_ports.pop(first.id)
    manager._assign_ports(first,chosen,'vLLM')
    assert manager._assigned_ports[first.id]==reserved


def test_follower_resolves_its_own_model_file_and_accepts_running_snapshot(launch_server):
    context=launch_server()
    server=context.server
    mi=distributed_instance().model_copy(update={'source':'model_scope','model_scope_model_id':'Qwen/fixture',
        'local_path':None,'resolved_path':'/main/cache/model','state':ModelInstanceStateEnum.RUNNING})
    server._worker=server._worker.model_copy(update={'id':2})
    file=SimpleNamespace(model_source_index=mi.model_source_index,state=ModelFileStateEnum.READY,resolved_paths=['/follower/cache/model'])
    server._clientset.model_files=SimpleNamespace(list=lambda **kwargs:SimpleNamespace(items=[file]))
    assert server._stop_when_starting(Event(type=EventType.CREATED,data=mi))
    assert server._model_path=='/follower/cache/model'


def test_manual_allowed_workers_can_supply_a_smaller_requested_count(tmp_path,config_stub):
    async def check():
        chosen=model(gpu_selector={'gpu_ids':[f'worker-{i}:cuda:0' for i in (1,2,3)],'gpus_per_replica':2})
        selector=VLLMResourceFitSelector(Config(data_dir=str(tmp_path)),chosen,[])
        result=await selector.select_candidates([worker(i,sizes=(24,)) for i in (1,2,3)])
        assert len(result)==1 and len(result[0].subordinate_workers)==1
    asyncio.run(check())


@pytest.mark.parametrize('kind',['cuda','cann'])
def test_distributed_communication_preserves_user_interface_overrides(launch_server,kind):
    context=launch_server(kind=kind,model_changes={'env':{'GLOO_SOCKET_IFNAME':'user-gloo','HCCL_IF_IP':'10.1.0.2'}})
    server=context.server
    server._model_instance=distributed_instance()
    server._model_instance.gpu_type=kind
    metadata=server._model_instance.get_deployment_metadata(1)
    env=server._get_configured_env(True,metadata)
    assert env['GLOO_SOCKET_IFNAME']=='user-gloo' and env['HCCL_IF_IP']=='10.1.0.2'


def test_reaping_stops_own_workload_and_removes_numbered_logs(tmp_path,monkeypatch):
    manager=ServeManager(lambda:2,lambda:None,Config(data_dir=str(tmp_path)))
    mi=distributed_instance()
    path=Path(manager._serve_log_dir)/f'{mi.id}.container.ray-head.0.log'
    path.write_text('fixture')
    deleted=[]
    monkeypatch.setattr(serve_module,'delete_workload',deleted.append)
    manager._stop_model_instance(mi,delete_logs=True)
    assert deleted==[mi.name+'-f0'] and not path.exists()
