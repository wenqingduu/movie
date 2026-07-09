#!/usr/bin/env python3
import argparse
import signal
import sys
import time

import torch


def mib(value: int) -> float:
    return value / 1024 / 1024


def allocate_to_fraction(device: torch.device, fraction: float, chunk_mib: int) -> list[torch.Tensor]:
    free, total = torch.cuda.mem_get_info(device)
    target_used = int(total * fraction)
    current_used = total - free
    to_allocate = max(0, target_used - current_used)

    print(
        f"GPU memory total={mib(total):.0f} MiB current={mib(current_used):.0f} MiB "
        f"target={mib(target_used):.0f} MiB allocate={mib(to_allocate):.0f} MiB",
        flush=True,
    )

    tensors: list[torch.Tensor] = []
    remaining = to_allocate
    chunk_bytes = chunk_mib * 1024 * 1024
    while remaining > 0:
        size = min(remaining, chunk_bytes)
        tensor = torch.empty(size, dtype=torch.uint8, device=device)
        tensor.fill_(1)
        tensors.append(tensor)
        remaining -= size
        used = total - torch.cuda.mem_get_info(device)[0]
        print(f"allocated chunk, GPU memory used={mib(used):.0f} MiB", flush=True)

    return tensors


def burn_gpu(device: torch.device, utilization: float, period_s: float, matrix_size: int) -> None:
    busy_s = period_s * utilization
    idle_s = max(0.0, period_s - busy_s)

    a = torch.randn((matrix_size, matrix_size), dtype=torch.float16, device=device)
    b = torch.randn((matrix_size, matrix_size), dtype=torch.float16, device=device)
    c = torch.empty((matrix_size, matrix_size), dtype=torch.float16, device=device)
    torch.cuda.synchronize(device)

    print(
        f"Starting GPU burn: target_util={utilization * 100:.0f}% "
        f"period={period_s:.2f}s busy={busy_s:.2f}s idle={idle_s:.2f}s",
        flush=True,
    )

    while True:
        start = time.monotonic()
        end = start + busy_s
        while time.monotonic() < end:
            torch.mm(a, b, out=c)
            torch.cuda.synchronize(device)

        elapsed = time.monotonic() - start
        sleep_for = max(0.0, period_s - elapsed)
        if sleep_for:
            time.sleep(sleep_for)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hold GPU memory and utilization near target fractions.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory", type=float, default=0.70, help="Target fraction of total GPU memory.")
    parser.add_argument("--util", type=float, default=0.70, help="Target GPU utilization duty cycle.")
    parser.add_argument("--period", type=float, default=1.0, help="Duty-cycle period in seconds.")
    parser.add_argument("--chunk-mib", type=int, default=512)
    parser.add_argument("--matrix-size", type=int, default=2048)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 1

    stop = False

    def handle_signal(signum, frame):
        nonlocal stop
        stop = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using {torch.cuda.get_device_name(device)} on cuda:{args.gpu}", flush=True)

    holders = allocate_to_fraction(device, args.memory, args.chunk_mib)
    torch.cuda.synchronize(device)

    try:
        burn_gpu(device, args.util, args.period, args.matrix_size)
    except KeyboardInterrupt:
        print("Stopping and releasing GPU memory...", flush=True)
        holders.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        return 0 if stop else 130


if __name__ == "__main__":
    raise SystemExit(main())
