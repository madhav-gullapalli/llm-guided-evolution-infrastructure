"""Exercise the actual async batch worker without loading model weights."""
import ast
import asyncio
from pathlib import Path
import threading
import time

import pytest


def worker(pipeline, batch_size=8):
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'server.py').read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMModel')
    methods = [n for n in model.body if isinstance(n, ast.AsyncFunctionDef)]
    cls = ast.ClassDef(name='Worker', bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = dict(asyncio=asyncio, time=time, BATCH_SIZE=batch_size, BATCH_WAIT_TIME=.01)
    exec(compile(ast.fix_missing_locations(ast.Module([cls], [])), 'server.py', 'exec'), namespace)
    instance = namespace['Worker']()
    instance.pipeline = pipeline
    instance.request_queue = asyncio.Queue()
    instance.batch_lock = asyncio.Lock()
    instance.batch_task = None
    instance.is_processing = False
    instance.batch_id = instance.request_id = 0
    return instance


def request(prompt, temperature=.7, top_p=.9, tokens=100):
    return dict(prompt=prompt, temperature=temperature, top_p=top_p, max_new_tokens=tokens)


async def stop(instance):
    instance.batch_task.cancel()
    await instance.batch_task


def test_batch_uses_one_call_with_reference_sampling_policy():
    calls = []

    def pipeline(prompts, **settings):
        calls.append((prompts, settings))
        return [[dict(generated_text=p)] for p in prompts]

    async def scenario():
        instance = worker(pipeline)
        results = await asyncio.wait_for(asyncio.gather(
            instance.generate(request('a', temperature=.2, top_p=.1)),
            instance.generate(request('b', temperature=.7, tokens=200)),
            instance.generate(request('c', temperature=.3)),
        ), 2)
        await asyncio.wait_for(instance.request_queue.join(), 1)
        await stop(instance)
        assert [r['generated_text'] for r in results] == ['a', 'b', 'c']
        assert all(r['batch_size'] == 3 for r in results)

    asyncio.run(scenario())
    assert calls == [(['a', 'b', 'c'], dict(max_new_tokens=200, temperature=.2, top_p=.1))]


def test_requests_queue_during_inference_and_cancelled_client_does_not_break_batch():
    started, release = threading.Event(), threading.Event()
    calls = []

    def pipeline(prompts, **settings):
        calls.append(prompts)
        if len(calls) == 1:
            started.set()
            assert release.wait(2)
        return [[dict(generated_text=p)] for p in prompts]

    async def scenario():
        instance = worker(pipeline, batch_size=2)
        first = asyncio.create_task(instance.generate(request('a')))
        second = asyncio.create_task(instance.generate(request('b')))
        try:
            async def wait_started():
                while not started.is_set():
                    await asyncio.sleep(.001)
            await asyncio.wait_for(wait_started(), 1)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            third = asyncio.create_task(instance.generate(request('c')))
            await asyncio.sleep(.01)
            assert instance.request_queue.qsize() == 1
            release.set()
            results = await asyncio.wait_for(asyncio.gather(second, third), 2)
            assert [r['generated_text'] for r in results] == ['b', 'c']
            await asyncio.wait_for(instance.request_queue.join(), 1)
        finally:
            release.set()
            await stop(instance)

    asyncio.run(scenario())
    assert calls == [['a', 'b'], ['c']]


@pytest.mark.parametrize('failure', ['exception', 'short_results'])
def test_failed_batch_releases_clients_and_worker_processes_next_batch(failure):
    count = 0

    def pipeline(prompts, **settings):
        nonlocal count
        count += 1
        if count == 1:
            if failure == 'exception':
                raise RuntimeError('inference failed')
            return []
        return [[dict(generated_text=p)] for p in prompts]

    async def scenario():
        instance = worker(pipeline, batch_size=2)
        results = await asyncio.wait_for(asyncio.gather(
            instance.generate(request('a')), instance.generate(request('b')),
            return_exceptions=True), 2)
        assert all(isinstance(r, RuntimeError) for r in results)
        result = await asyncio.wait_for(instance.generate(request('c')), 2)
        assert result['generated_text'] == 'c'
        await asyncio.wait_for(instance.request_queue.join(), 1)
        await stop(instance)

    asyncio.run(scenario())
