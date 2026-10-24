"""Compose Python modules through explicit exports, preserving source globals."""
from pathlib import Path
import keyword


def compose_sources(sources, exports, output_file):
    """Write a module from N source paths; exports maps public name to source index.

    Source paths are absolute so jobs need no extra import-path setup. Like normal
    imports, dependencies of each source must be available at runtime.
    """
    paths = [str(Path(source).resolve()) for source in sources]
    if not paths:
        raise ValueError("At least one source is required")
    for name, owner in exports.items():
        if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_coev_"):
            raise ValueError(f"Invalid/reserved export name: {name}")
        if not isinstance(owner, int) or not 0 <= owner < len(paths):
            raise ValueError(f"Unknown source index for {name}: {owner}")
    lines = [
        "# Generated composition; source modules retain their own globals.",
        "import importlib.util as _coev_importlib",
        "import sys as _coev_sys", "",
        "def _coev_load(index, path):",
        "    name = __name__ + '_source_' + str(index)",
        "    spec = _coev_importlib.spec_from_file_location(name, path)",
        "    module = _coev_importlib.module_from_spec(spec)",
        "    _coev_sys.modules[name] = module",
        "    try:", "        spec.loader.exec_module(module)",
        "    except BaseException:",
        "        _coev_sys.modules.pop(name, None)", "        raise",
        "    return module", "",
    ]
    lines += [f"_coev_source_{i} = _coev_load({i}, {path!r})"
              for i, path in enumerate(paths)]
    lines += [f"{name} = getattr(_coev_source_{owner}, {name!r})"
              for name, owner in exports.items()]
    lines.append(f"__all__ = {list(exports)!r}")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output
