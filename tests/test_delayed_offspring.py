"""Failed LLM offspring must revert to an existing parent before mutation."""
import pytest


@pytest.mark.parametrize('operation', ['mate', 'mutate'])
@pytest.mark.parametrize('submitted,completed,output_exists', [
    (False, False, False), (True, False, False),
    (True, True, False), (True, True, True),
])
def test_delayed_offspring_recovery(monkeypatch, tmp_path, operation,
                                   submitted, completed, output_exists):
    import run_improved as runner

    monkeypatch.setattr(runner, 'DELAYED_CHECK', True)
    monkeypatch.setattr(runner, 'VARIANT_DIR', str(tmp_path))
    parent_data = {'status': 'evaluated', 'fitness': (1, 2)}
    monkeypatch.setattr(runner, 'GLOBAL_DATA', {
        'parent': parent_data.copy(),
        'child': {'status': 'DELAYED_CHECK', 'sub_flag': submitted, 'job_id': 123},
    })
    monkeypatch.setattr(runner, 'LINKED_GENES', {'child': 'parent'})
    (tmp_path / f'{runner.MODEL}_parent.py').write_text('# parent\n')
    if output_exists:
        (tmp_path / f'{runner.MODEL}_child.py').write_text('# child\n')
    calls = []

    def check(job_id, **kwargs):
        calls.append(job_id)
        return completed

    monkeypatch.setattr(runner, 'check4job_completion', check)
    check_offspring = getattr(runner, f'delayed_{operation}_check')
    offspring = check_offspring([runner.creator.Individual(['child'])])
    success = submitted and completed and output_exists
    assert offspring[0][0] == ('child' if success else 'parent')
    assert (tmp_path / f'{runner.MODEL}_{offspring[0][0]}.py').is_file()
    assert ('child' in runner.GLOBAL_DATA) == success
    if not success:
        assert runner.GLOBAL_DATA['parent'] == parent_data
    assert calls == ([123] if submitted else [])
