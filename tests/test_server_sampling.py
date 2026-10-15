"""Exercise batch sampling without loading the GPU model or HTTP server."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace


def test_batch_preserves_each_requests_sampling_settings():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'server.py').read_text())
    processor = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.AsyncFunctionDef) and n.name == '_batch_processor')
    statements = next(n.body for n in ast.walk(processor)
                      if isinstance(n, ast.Try) and any(
                          isinstance(s, ast.Assign) and any(
                              isinstance(t, ast.Name) and t.id == 'groups' for t in s.targets)
                          for s in n.body))
    start = next(i for i, n in enumerate(statements)
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == 'groups')
    stop = next(i for i, n in enumerate(statements)
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == 'response_time')
    wrapper = ast.parse('async def generate(self, batch):\n    pass\n').body[0]
    wrapper.body = statements[start:stop] + [ast.Return(ast.Name('results', ast.Load()))]
    module = ast.fix_missing_locations(ast.Module([wrapper], []))
    async def to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    namespace = {'asyncio': SimpleNamespace(to_thread=to_thread)}
    exec(compile(module, 'server_sampling', 'exec'), namespace)
    calls = []

    def pipeline(prompts, **settings):
        calls.append((prompts, settings))
        return [(prompt, settings.copy()) for prompt in prompts]

    requests = [dict(prompt=p, max_new_tokens=100, temperature=t, top_p=n)
                for p, t, n in [('a', .2, .1), ('b', .7, .9), ('c', .2, .1)]]
    results = asyncio.run(namespace['generate'](SimpleNamespace(pipeline=pipeline), requests))
    assert len(calls) == 2
    assert [r[0] for r in results] == ['a', 'b', 'c']
    for result, request in zip(results, requests):
        assert result[1] == {k: request[k] for k in ('max_new_tokens', 'temperature', 'top_p')}
