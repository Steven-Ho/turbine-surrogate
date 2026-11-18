"""Two-node GPU all-reduce probe, launched by torchrun on each node."""

import datetime
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=180),
    )
    rank = dist.get_rank()
    world = dist.get_world_size()

    info = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "visible_devices": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(local_rank),
        "torch": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(info), flush=True)

    ranks = [None] * world
    dist.all_gather_object(ranks, info)
    hosts = {item["hostname"] for item in ranks}
    assert len(hosts) == 2, f"Expected two nodes, got {hosts}"

    results = []
    for mib in (1, 16, 64, 256):
        count = mib * 1024 * 1024 // 4
        x = torch.empty(count, device=device, dtype=torch.float32)

        x.fill_(rank + 1)
        dist.all_reduce(x)
        expected = world * (world + 1) / 2
        assert torch.all(x == expected).item(), (
            f"all_reduce correctness failed: rank={rank}, payload={mib}"
        )

        x.zero_()
        for _ in range(5):
            dist.all_reduce(x)

        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        iterations = 30
        for _ in range(iterations):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / iterations

        elapsed = torch.tensor(seconds, device=device, dtype=torch.float32)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        seconds = elapsed.item()
        results.append({
            "payload_MiB": mib,
            "all_reduce_ms": seconds * 1000,
            "payload_GB_per_second": (
                x.numel() * x.element_size() / seconds / 1e9
            ),
        })

    output = {
        "backend": str(dist.get_backend()),
        "world_size": world,
        "ranks": ranks,
        "results": results,
        "note": "Payload throughput; not physical link or bus bandwidth.",
    }
    if rank == 0:
        path = Path(os.environ["PDE_FM_PROBE_DIR"]) / "result.json"
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"PDE_FM_MULTINODE_RANK_OK rank={rank}", flush=True)


if __name__ == "__main__":
    main()
