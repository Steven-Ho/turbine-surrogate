"""Extract a leakage-free mesh+BC -> nodal uvwpT sample from case4."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import h5py
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/data/hesc/测试数据/case4")
CGNS_TO_VTK = {5:5, 7:9, 10:10, 12:14, 14:13}
NODE_COUNTS = {5:3, 7:4, 10:4, 12:5, 14:6}
FLAGS = {"stagnation":1, "inlet":2, "outlet":4, "wall":8, "symm":16, "sliding":32}


def scalar(text):
    text = text.strip()
    if text.startswith(("'", '"')): return text[1:-1]
    if text.lower() in [".true.", ".false."]: return text.lower() == ".true."
    try:
        value = float(text.replace("D", "E").replace("d", "e"))
        return int(value) if value.is_integer() else value
    except ValueError: return text


def namelists(text):
    result = []
    for match in re.finditer(r"^\s*&(\w+)\s*\n(.*?)^\s*/\s*$", text, re.M | re.S):
        body = match.group(2)
        assignments = list(re.finditer(r"([A-Za-z_]\w*)\s*=", body))
        values = {}
        for index, item in enumerate(assignments):
            end = assignments[index+1].start() if index+1 < len(assignments) else len(body)
            raw = body[item.end():end].strip().rstrip(",")
            if raw.startswith(("'", '"')): value = scalar(raw)
            else:
                parts = next(csv.reader(io.StringIO(raw.replace("\n", " ")), skipinitialspace=True))
                parsed = [scalar(part) for part in parts]
                value = parsed[0] if len(parsed)==1 else parsed
            values[item.group(1)] = value
        result.append({"section": match.group(1), "values": values})
    return result


def boundary_faces(section):
    code = int(section[" data"][0]); values = section["ElementConnectivity/ data"][...].astype(np.int64) - 1
    if code != 20:
        count = NODE_COUNTS[code]
        assert len(values) % count == 0
        return values, np.arange(0, len(values)+1, count, dtype=np.int64)
    # MIXED contains a CGNS type code before every face, not a node ID.
    raw = section["ElementConnectivity/ data"][...]
    rows=[]; cursor=0
    while cursor < len(raw):
        count = NODE_COUNTS[int(raw[cursor])]
        rows.append(np.asarray(raw[cursor+1:cursor+1+count], dtype=np.int64)-1)
        cursor += count+1
    offsets = np.r_[0,np.cumsum([len(row) for row in rows])]
    return np.concatenate(rows), offsets


def verify_volume(section, connectivity, offsets, types, node_offset):
    code = int(section[" data"][0])
    raw = section["ElementConnectivity/ data"]
    count = len(types)
    assert code == 20, "This exporter expects the observed MIXED volume sections"
    assert len(raw) == int(offsets[-1]) + count
    lookup = np.zeros(256, dtype=np.int64)
    for cgns,vtk_type in CGNS_TO_VTK.items(): lookup[vtk_type] = cgns
    for start in range(0,count,200000):
        end=min(start+200000,count); first=int(offsets[start]); last=int(offsets[end])
        segment = raw[first+start:last+end]
        positions = offsets[start:end]-first + np.arange(end-start)
        np.testing.assert_array_equal(segment[positions],lookup[types[start:end]])
        keep = np.ones(len(segment),dtype=bool);keep[positions]=False
        # CGNS and VTK may use different local corner order conventions.
        # Check each cell's exact node membership and type, not just counts.
        source_nodes=segment[keep]-1
        vtk_nodes=connectivity[first:last]-node_offset
        local_starts=offsets[start:end]-first
        for vtk_type in np.unique(types[start:end]):
            selected=np.flatnonzero(types[start:end]==vtk_type)
            width=NODE_COUNTS[int(lookup[vtk_type])]
            indices=local_starts[selected,None]+np.arange(width)[None,:]
            np.testing.assert_array_equal(np.sort(source_nodes[indices],axis=1),np.sort(vtk_nodes[indices],axis=1))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--output",type=Path,default=ROOT/"data/stator_case_000/sample.h5")
    args=parser.parse_args()
    source=args.source.resolve(); case=source/"adhoc_runs/main/node_cfd_cfd04/solve/execute/case"
    mesh=source/"adhoc_runs/main/node_geo_geo041/mesh/rotor_surface_3d.cgns"
    solution=case/"result/ROTOR_FLUID_0512.vtk"
    metadata=ROOT/"data/stator_case_000";metadata.mkdir(parents=True,exist_ok=True)
    sections=namelists((case/"AIflow.ctl").read_text(encoding="utf-8-sig"))
    controls=json.loads((case/"AIflow.json").read_text(encoding="utf-8-sig"))
    boundaries={item["values"]["name"]:item["values"] for item in sections if item["section"]=="boundary"}
    parameters={"inlet_total_pressure_pa":boundaries["inlet"]["p"],
                "inlet_total_temperature_k":boundaries["inlet"]["t"],
                "outlet_static_pressure_pa":boundaries["outlet"]["p"],
                "cold_up_mass_flow_kg_s":boundaries["cold_in_up"]["massflux"],
                "cold_up_temperature_k":boundaries["cold_in_up"]["t"],
                "cold_down_mass_flow_kg_s":boundaries["cold_in_down"]["massflux"],
                "cold_down_temperature_k":boundaries["cold_in_down"]["t"]}
    gui=json.loads((case/"GUIInfo.json").read_text(encoding="utf-8-sig"))
    for items in gui["CFDSimulationData"]["DomainData"]["DomainBoundary"].values():
        for item in items:
            if item.get("PatchList") in ["cold_in_up","cold_in_down"]:
                assert item["MomentData"][0]["MomentumData"]["MassFlowRate"]==.18
    bc={"parameters":parameters,"boundary_conditions":boundaries,
        "physics_sections":[item for item in sections if item["section"] in ["model","fluid","kemodel","species"]],
        "mesh_controls":controls["mesh"],"materials":controls["materials"],
        "semantics":{"Velocity":"absolute/stationary-frame velocity; all configured RPM values are zero",
                     "p":"Static_Pressure, not Total_Pressure", "T":"Temperature, not Total_Temperature",
                     "stagnation":"total inlet p0,T0; not nodal static pressure/temperature Dirichlet values",
                     "cold_inlets":"mass-flow normal inlet, not u=v=w=0 Dirichlet",
                     "Neumann_t_zero":"zero thermal gradient/flux setting; not zero kelvin wall",
                     "symm":"source really declares symmetry; no periodic pairing inferred from per_* names",
                     "interfaces":"retain domain pairs; no unsupported conformal node merge"}}
    (metadata/"boundary_conditions.json").write_text(json.dumps(bc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Reading complete volume solution",flush=True)
    reader=vtk.vtkUnstructuredGridReader();reader.SetFileName(str(solution));reader.ReadAllScalarsOn();reader.ReadAllVectorsOn();reader.ReadAllFieldsOn();reader.Update()
    grid=reader.GetOutput();positions=vtk_to_numpy(grid.GetPoints().GetData())
    conn=vtk_to_numpy(grid.GetCells().GetConnectivityArray());offsets=vtk_to_numpy(grid.GetCells().GetOffsetsArray());types=vtk_to_numpy(grid.GetCellTypesArray())
    names=["u","v","w","p","T"]
    velocity=vtk_to_numpy(grid.GetPointData().GetArray("Velocity"))
    pressure=vtk_to_numpy(grid.GetPointData().GetArray("Static_Pressure"))
    temperature=vtk_to_numpy(grid.GetPointData().GetArray("Temperature"))
    targets=np.column_stack([velocity,pressure,temperature]).astype(np.float32)
    assert targets.shape==(grid.GetNumberOfPoints(),5) and np.isfinite(targets).all()
    summary={"sample_id":"stator_case_000","source_case":str(source),"mesh_source":str(mesh),"solution_source":str(solution),
             "nodes":len(positions),"volume_cells":len(types),"target_names":names,"target_location":"point",
             "parameters":parameters,"independent_operating_cases":1,"eligible_for_generalization_benchmark":False,
             "regions":[],"boundary_flags":FLAGS,"boundaries":[],"source_input_mesh_verified_against_output":True,"cell_verification":"exact per-cell node sets and types; local CGNS/VTK corner ordering may differ"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(mesh,"r") as original,h5py.File(args.output,"w") as out:
        def write(group,name,array):
            return group.create_dataset(name,data=array,compression="lzf",shuffle=True)
        inp=out.create_group("inputs");m=inp.create_group("mesh")
        write(m,"positions",positions.astype(np.float32));write(m,"cell_connectivity",conn.astype(np.int32));write(m,"cell_offsets",offsets);write(m,"cell_types",types)
        flags=np.zeros(len(positions),np.uint16);region_ids=np.zeros(len(positions),np.uint8)
        bg=inp.create_group("boundaries");n0=c0=0
        for region_id,domain in enumerate(controls["mesh"]["domains"]):
            name=domain["name"];zone=original["Base/"+name];size=zone[" data"][...].reshape(-1);n,nc=int(size[0]),int(size[1])
            xyz=np.column_stack([zone["GridCoordinates/Coordinate"+axis+"/ data"][...].reshape(-1) for axis in "XYZ"]).astype(np.float32)
            np.testing.assert_array_equal(positions[n0:n0+n],xyz)
            local_offsets=offsets[c0:c0+nc+1]-offsets[c0]
            verify_volume(zone[name],conn[int(offsets[c0]):int(offsets[c0+nc])],local_offsets,types[c0:c0+nc],n0)
            region_ids[n0:n0+n]=region_id
            summary["regions"].append({"id":region_id,"name":name,"node_offset":n0,"nodes":n,"cell_offset":c0,"cells":nc,"rpm":domain["rotation"]})
            for item in domain["boundary_list"]+domain["interface_list"]:
                patch=item["name"]
                flat,face_offsets=boundary_faces(zone[patch]);ids=np.unique(flat)+n0
                assert flat.min()>=0 and flat.max()<n
                condition=boundaries.get(patch)
                if condition is None:
                    condition=next(value for value in boundaries.values() if value.get("name2")==patch)
                kind=condition["kind"];flags[ids]|=FLAGS[kind]
                g=bg.require_group(name).create_group(patch)
                write(g,"node_ids",ids.astype(np.int32));write(g,"face_connectivity",(flat+n0).astype(np.int32));write(g,"face_offsets",face_offsets)
                g.attrs["bc_json"]=json.dumps(condition);g.attrs["region_id"]=region_id
                summary["boundaries"].append({"region":name,"name":patch,"kind":kind,"nodes":len(ids),"faces":len(face_offsets)-1})
            n0+=n;c0+=nc;print("Verified CGNS inputs and result topology:",name,n,nc,flush=True)
        assert n0==len(positions) and c0==len(types)
        write(m,"node_region_ids",region_ids);write(inp,"boundary_flags",flags)
        inp.create_dataset("global_bc_values",data=np.asarray(list(parameters.values()),np.float64))
        inp.attrs["global_bc_names"]=json.dumps(list(parameters))
        inp.attrs["boundary_conditions_json"]=json.dumps(bc)
        y=out.create_group("targets");write(y,"uvwpT",targets);y.attrs["field_names"]=json.dumps(names)
        # Source residuals are quality metadata, never model inputs.
        lines=(case/"monitor/residual.csv").read_text(encoding="utf-8-sig").splitlines()
        residual_header=lines[0].split(",");last=lines[-1].split(",")
        quality={"label_status":"needs_solver_quality_review", "all_five_target_channels_finite":True,
                 "temperature_at_200K_nodes":int(np.sum(temperature<=200.001)),
                 "temperature_at_5000K_nodes":int(np.sum(temperature>=4999.999)),
                 "temperature_lower_limit_k":200,"temperature_upper_limit_k":5000,
                 "last_residual_record":dict(zip(residual_header,last)),
                 "warning":"Single saved operating point; finite fields do not establish solver convergence or generalization."}
        summary["quality"]=quality
        summary["target_ranges"]={name:[float(targets[:,i].min()),float(targets[:,i].max())] for i,name in enumerate(names)}
        out.attrs["sample_metadata_json"]=json.dumps(summary)
    summary["hdf5_path"]=str(args.output.resolve());summary["hdf5_bytes"]=args.output.stat().st_size
    summary["hdf5_sha256"]=hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary["source_files"]={str(path):{"bytes":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns} for path in [mesh,solution,case/"AIflow.ctl",case/"AIflow.json",case/"GUIInfo.json"]}
    (metadata/"manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:summary[key] for key in ["nodes","volume_cells","hdf5_path","hdf5_bytes","target_ranges","quality"]},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
