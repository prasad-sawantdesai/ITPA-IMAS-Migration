"""Shared SimDB import/data-access modules for the notebooks.

SimDB is used only to *find* pulses, by their `dataset`/`machine` labels. The values themselves are
read from each pulse's IMAS HDF5 output (`load`), because the SimDB develop branch keeps a numeric
metadata array only as its {"min", "max"} range (simdb/json.py, CustomEncoder), which loses the
per-time-slice data. Reading the HDF5 files works the same with the old (<= 0.15) and the current
develop SimDB local database.
"""

import re
import urllib.parse
from typing import Any, Optional

import imas
import numpy as np
import numpy.typing as npt
from imas.ids_defs import EMPTY_FLOAT, EMPTY_INT
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure

from simdb.config.config import Config
from simdb.database import get_local_db
from simdb.query import QueryType

# Catalogue labels taken from the SimDB entry; everything else comes from the HDF5 files.
CATALOGUE_KEYS = ("dataset", "machine", "pulse")

_loaded: dict[str, dict] = {}  # sim UUID -> load() result, so repeated passes don't re-read HDF5


def get_db():
    """Connect to the local SimDB."""
    return get_local_db(Config())


def guard(x: npt.ArrayLike) -> np.ndarray:
    """Replace the IMAS empty-float/empty-int sentinels with NaN."""
    a = np.asarray(x, dtype=float)
    is_empty = (np.abs(a) >= abs(EMPTY_FLOAT)) | (a == EMPTY_INT)
    return np.where(is_empty, np.nan, a)


def path(md: dict, *keys: str, n: int) -> np.ndarray:
    """Walk a nested meta_dict path"""
    node = md
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return np.full(n, np.nan)
        node = node[k]
    return guard(node)


def temp(md: dict, *names: str, n: int) -> np.ndarray:
    """First matching temporary-IDS quantity: db_variable.<name> or standard_name.<name>."""
    for bucket in ("db_variable", "standard_name"):
        for name in names:
            v = md.get(bucket, {}).get(name)
            if v is not None:
                return guard(v)
    return np.full(n, np.nan)


def temp_str(md: dict, *names: str, n: int) -> np.ndarray:
    """Like temp(), but for string-valued temporary-IDS quantities."""
    for bucket in ("db_variable", "standard_name"):
        for name in names:
            v = md.get(bucket, {}).get(name)
            if v is not None:
                return np.asarray(v, dtype=object)
    return np.full(n, "", dtype=object)


def path_str(md: dict, *keys: str, n: int) -> np.ndarray:
    """Like path(), but for string-valued quantities. Broadcasts a shot-scalar to length n."""
    node = md
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return np.full(n, "", dtype=object)
        node = node[k]
    arr = np.asarray(node, dtype=object)
    if arr.ndim == 0:
        return np.full(n, arr.item(), dtype=object)
    return arr


def query_selected(db, dataset: str, selec_key: str, machine: Optional[str] = None, value: str = "1") -> list[Any]:
    """Simulations for `dataset` (optionally restricted to `machine`) with at least one
    time-slice where db_variable.<selec_key> == value.

    The flag is tested on the loaded per-slice values rather than in the SimDB query, because SimDB
    versions differ in how they store and compare metadata arrays.
    """
    try:
        target: Any = float(value)
    except ValueError:
        target = None
    selected = []
    for sim in query_dataset(db, dataset, machine=machine):
        md = load(sim)
        if target is None:
            hit = np.any(temp_str(md, selec_key, n=0) == value)
        else:
            hit = np.any(temp(md, selec_key, n=0) == target)
        if hit:
            selected.append(sim)
    return selected


def query_dataset(db, dataset: str, machine: Optional[str] = None) -> list[Any]:
    """All simulations for `dataset` (optionally restricted to `machine`), unfiltered by any
    selection flag."""
    constraints = [("dataset", dataset, QueryType.EQ)]
    if machine is not None:
        constraints.append(("machine", machine, QueryType.EQ))
    return db.query_meta(constraints)


# ---------------------------------------------------------------------------
# Reading a pulse's values from its HDF5 output
# ---------------------------------------------------------------------------


def load(sim) -> dict:
    """A pulse's data as a nested dict, in the same shape as `sim.meta_dict()`, but read from HDF5.

    Keys follow the `summary` IDS (e.g. md["global_quantities"]["ip"]["value"]), plus the
    `temporary` IDS quantities under md["db_variable"] / md["standard_name"] and the catalogue
    labels `dataset`, `machine`, `pulse`. Arrays keep every time-slice. Results are cached per
    simulation; call `clear_cache()` after re-running the migration.
    """
    key = str(sim.uuid)
    if key not in _loaded:
        _loaded[key] = _read_pulse(sim)
    return _loaded[key]


def clear_cache() -> None:
    """Forget every pulse read by `load`."""
    _loaded.clear()


def hdf5_path(sim) -> str:
    """Directory of the pulse's IMAS HDF5 output, from the URI that SimDB recorded for it."""
    for output in sim.outputs:
        uri = str(output.uri)
        match = re.search(r"[?&;]path=([^;&#]+)", uri)
        if uri.startswith("imas:") and match:
            return urllib.parse.unquote(match.group(1))
    raise ValueError(f"{sim.alias}: SimDB records no IMAS HDF5 output for this simulation")


def _read_pulse(sim) -> dict:
    meta = sim.meta_dict()
    pulse_dir = hdf5_path(sim)
    with imas.DBEntry(f"imas:hdf5?path={pulse_dir}", "r") as entry:
        md = _to_dict(entry.get("summary", autoconvert=False))
        if entry.list_all_occurrences("temporary"):
            variables = _temporary_variables(entry.get("temporary", autoconvert=False))
        else:
            variables = _variables_from_meta(sim, meta)
    for k in CATALOGUE_KEYS:
        if k in meta:
            md.setdefault(k, meta[k])
    # Split as idsmigration's make_manifest did: names SimDB files under standard_name.* stay there.
    standard = meta.get("standard_name", {})
    for name, value in variables.items():
        bucket = "standard_name" if name in standard else "db_variable"
        md.setdefault(bucket, {})[name] = value
    return md


def _to_dict(node: IDSStructure) -> dict:
    """Filled leaves of an IDS (sub)tree as a nested dict: arrays as np.ndarray, 0D as scalars."""
    out: dict[str, Any] = {}
    for child in node.iter_nonempty_():
        name = child.metadata.name
        if isinstance(child, IDSStructArray):
            out[name] = [_to_dict(el) for el in child]
        elif isinstance(child, IDSStructure):
            out[name] = _to_dict(child)
        elif child.metadata.ndim:
            out[name] = np.asarray(child.value)
        else:
            out[name] = child.value
    return out


def _temporary_variables(temp_ids: IDSStructure) -> dict:
    """{identifier/name: value} for every filled entry of the temporary IDS buckets.

    constant_* entries hold `value` directly; dynamic_* entries hold it in `value/data`.
    """
    result: dict[str, Any] = {}
    for bucket in temp_ids.iter_nonempty_():
        if not isinstance(bucket, IDSStructArray):
            continue
        dynamic = bucket.metadata.name.startswith("dynamic_")
        for el in bucket:
            name = str(el.identifier.name)
            leaf = el.value.data if dynamic else el.value
            if name and leaf.has_value:
                result[name] = np.asarray(leaf.value) if leaf.metadata.ndim else leaf.value
    return result


def _variables_from_meta(sim, meta: dict) -> dict:
    """Temporary quantities from SimDB metadata, for pulses migrated without a temporary.h5.

    Old SimDB keeps full arrays here; the develop branch keeps numeric arrays only as a
    {"min", "max"} range, which cannot be turned back into per-slice values.
    """
    result: dict[str, Any] = {}
    ranged = []
    for bucket in ("db_variable", "standard_name"):
        for name, value in meta.get(bucket, {}).items():
            if isinstance(value, dict) and set(value) == {"min", "max"}:
                ranged.append(name)
            else:
                result[name] = value
    if ranged:
        raise RuntimeError(
            f"{sim.alias}: no temporary IDS in {hdf5_path(sim)}, and SimDB stores only the min/max of "
            f"{', '.join(sorted(ranged))}. Re-run idsmigration --simdb for this dataset so that "
            "temporary.h5 is written next to summary.h5."
        )
    return result
