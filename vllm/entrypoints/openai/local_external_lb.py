#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import copy
import multiprocessing
import signal
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.process import BaseProcess
from typing import Any

import uvloop

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.system_utils import (
    decorate_logs,
    set_process_title,
    update_environment_variables,
)
from vllm.v1.utils import shutdown as shutdown_processes

logger = init_logger(__name__)

HEALTHCHECK_INTERVAL_S = 5.0
HEALTHCHECK_TIMEOUT_S = 5.0


def infer_local_external_lb_start_rank(args: argparse.Namespace) -> int:
    start_rank = getattr(args, "data_parallel_start_rank", None)
    if start_rank is not None:
        return start_rank

    node_rank = getattr(args, "node_rank", 0) or 0
    local_size = getattr(args, "data_parallel_size_local", 0) or 0
    return node_rank * local_size


def validate_local_external_lb_args(args: argparse.Namespace) -> None:
    if getattr(args, "grpc", False):
        raise ValueError(
            "Error: --data-parallel-local-external-lb does not support --grpc"
        )
    if args.uds is not None:
        raise ValueError(
            "Error: --data-parallel-local-external-lb does not support --uds"
        )
    if any((args.ssl_keyfile, args.ssl_certfile, args.ssl_ca_certs)):
        raise ValueError(
            "Error: --data-parallel-local-external-lb does not support HTTPS yet"
        )
    if args.api_server_count not in (None, 1):
        raise ValueError(
            "Error: --data-parallel-local-external-lb currently requires "
            "--api-server-count=1"
        )
    if args.data_parallel_rank is not None:
        raise ValueError(
            "Error: --data-parallel-local-external-lb manages child "
            "--data-parallel-rank values internally"
        )
    if args.data_parallel_external_lb or args.data_parallel_hybrid_lb:
        raise ValueError(
            "Error: --data-parallel-local-external-lb cannot be combined with "
            "--data-parallel-external-lb or --data-parallel-hybrid-lb"
        )
    if args.data_parallel_size < 2:
        raise ValueError(
            "Error: --data-parallel-local-external-lb requires --data-parallel-size > 1"
        )

    local_size = args.data_parallel_size_local
    if local_size is None or local_size < 2:
        raise ValueError(
            "Error: --data-parallel-local-external-lb requires "
            "--data-parallel-size-local >= 2"
        )
    if local_size > args.data_parallel_size:
        raise ValueError(
            "Error: --data-parallel-size-local cannot exceed --data-parallel-size"
        )
    if args.data_parallel_size % local_size != 0:
        raise ValueError(
            "Error: --data-parallel-size must be divisible by "
            "--data-parallel-size-local"
        )

    start_rank = infer_local_external_lb_start_rank(args)
    if start_rank + local_size > args.data_parallel_size:
        raise ValueError(
            "Error: local supervised ranks would exceed --data-parallel-size"
        )

    admin_port = args.data_parallel_admin_port
    child_port_min = args.port
    child_port_max = args.port + local_size - 1
    if child_port_min <= admin_port <= child_port_max:
        raise ValueError(
            f"Error: --data-parallel-admin-port {admin_port} "
            f"overlaps with child rank ports {child_port_min}-{child_port_max}"
        )


def build_local_external_lb_child_args(
    args: argparse.Namespace, local_rank: int
) -> argparse.Namespace:
    child_args = copy.copy(args)
    child_args.port = args.port + local_rank
    child_args.data_parallel_rank = (
        infer_local_external_lb_start_rank(args) + local_rank
    )
    child_args.data_parallel_start_rank = None
    child_args.data_parallel_size_local = 1
    child_args.data_parallel_external_lb = True
    child_args.data_parallel_hybrid_lb = False
    child_args.data_parallel_local_external_lb = False
    child_args.data_parallel_admin_port = None
    child_args.api_server_count = 1
    return child_args


def _build_local_external_lb_child_env(
    args: argparse.Namespace, local_rank: int
) -> dict[str, str]:
    # set visible devices for the child process
    devices_per_rank = args.tensor_parallel_size * args.pipeline_parallel_size
    start = local_rank * devices_per_rank
    stop = start + devices_per_rank
    device_env = current_platform.device_control_env_var
    visible_devices = ",".join(
        str(current_platform.device_id_to_physical_device_id(idx))
        for idx in range(start, stop)
    )
    return {device_env: visible_devices}


def _child_base_url(args: argparse.Namespace, port: int) -> str:
    host = args.host or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    return f"http://{host}:{port}"


def _probe_endpoint(
    args: argparse.Namespace, port: int, path: str
) -> tuple[bool, str | None]:
    request = urllib.request.Request(_child_base_url(args, port) + path, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=HEALTHCHECK_TIMEOUT_S) as response:
            return response.status == HTTPStatus.OK, None
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


@dataclass
class LocalExternalLBChildStatus:
    local_rank: int
    data_parallel_rank: int
    port: int
    pid: int | None = None
    healthy: bool = False
    ready: bool = False
    exitcode: int | None = None
    last_error: str | None = None


class LocalExternalLBState:
    def __init__(self, children: list[LocalExternalLBChildStatus]):
        self._lock = threading.Lock()
        self._children = children
        self._shutting_down = False

    def update_children(self, children: list[LocalExternalLBChildStatus]) -> None:
        with self._lock:
            self._children = children

    def begin_shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True

    def is_healthy(self) -> bool:
        with self._lock:
            return not self._shutting_down and all(
                child.healthy and child.exitcode is None for child in self._children
            )

    def is_ready(self) -> bool:
        with self._lock:
            return not self._shutting_down and all(
                child.ready and child.exitcode is None for child in self._children
            )


class _LocalExternalLBHTTPRequestHandler(BaseHTTPRequestHandler):
    server: _LocalExternalLBHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        # handle aggregated health and readiness statuses
        if self.path == "/health":
            self._send_status(self.server.state.is_healthy())
            return
        if self.path in ("/ready", "/readyz"):
            self._send_status(self.server.state.is_ready())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_status(self, ok: bool) -> None:
        self.send_response(HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE)
        self.send_header("Content-Length", "0")
        self.end_headers()


class _LocalExternalLBHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        state: LocalExternalLBState,
    ) -> None:
        self.state = state
        super().__init__(server_address, _LocalExternalLBHTTPRequestHandler)


def _run_local_external_lb_child(
    child_args: argparse.Namespace, env_updates: dict[str, str]
) -> None:
    from vllm.entrypoints.openai.api_server import run_server

    rank = child_args.data_parallel_rank
    update_environment_variables(env_updates)
    set_process_title("ExternalLBRank", str(rank))
    decorate_logs(f"ExternalLBRank{rank}")
    uvloop.run(run_server(child_args))


class LocalExternalLBSupervisor:
    def __init__(self, args: argparse.Namespace):
        validate_local_external_lb_args(args)
        self.args = args
        self.admin_port = args.data_parallel_admin_port
        self.child_specs = [
            LocalExternalLBChildStatus(
                local_rank=local_rank,
                data_parallel_rank=infer_local_external_lb_start_rank(args)
                + local_rank,
                port=args.port + local_rank,
            )
            for local_rank in range(args.data_parallel_size_local)
        ]
        self.state = LocalExternalLBState(copy.deepcopy(self.child_specs))
        self.processes: list[BaseProcess] = []
        self._stop_requested = threading.Event()
        self._failed_process: BaseProcess | None = None
        self._admin_server: _LocalExternalLBHTTPServer | None = None
        self._admin_thread: threading.Thread | None = None

    def run(self) -> None:
        previous_handlers = {
            sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
        }

        def _handle_signal(signum, frame):
            logger.info(
                "Received signal %d, forwarding graceful termination to local "
                "external LB child ranks",
                signum,
            )
            self.state.begin_shutdown()
            self._stop_requested.set()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        try:
            self._start_admin_server()
            self._start_children()
            self._monitor_children()
        finally:
            self.state.begin_shutdown()
            self._stop_requested.set()
            self._shutdown_children()
            self._shutdown_admin_server()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

        if self._failed_process is not None:
            raise RuntimeError(
                f"Local external LB child exited unexpectedly: "
                f"{self._failed_process.name} "
                f"exit code {self._failed_process.exitcode}"
            )

    def _start_admin_server(self) -> None:
        host = self.args.host or "0.0.0.0"
        self._admin_server = _LocalExternalLBHTTPServer(
            (host, self.admin_port), self.state
        )
        self._admin_thread = threading.Thread(
            target=self._admin_server.serve_forever,
            daemon=True,
            name="local-external-lb-admin",
        )
        self._admin_thread.start()
        logger.info_once(
            "Started local external LB admin server on %s:%d",
            host,
            self.admin_port,
        )

    def _start_children(self) -> None:
        context = multiprocessing.get_context("spawn")
        for local_rank in range(self.args.data_parallel_size_local):
            child_args = build_local_external_lb_child_args(self.args, local_rank)
            child_env = _build_local_external_lb_child_env(self.args, local_rank)
            process = context.Process(
                target=_run_local_external_lb_child,
                name=f"ExternalLBRank_{child_args.data_parallel_rank}",
                args=(child_args, child_env),
            )
            process.start()
            self.processes.append(process)

    def _monitor_children(self) -> None:
        while not self._stop_requested.is_set():
            statuses: list[LocalExternalLBChildStatus] = []
            for spec, process in zip(self.child_specs, self.processes):
                status = copy.copy(spec)
                status.pid = process.pid
                status.exitcode = process.exitcode

                if process.exitcode is not None:
                    status.last_error = f"process exited with code {process.exitcode}"
                    self._failed_process = process
                else:
                    status.healthy, status.last_error = _probe_endpoint(
                        self.args, status.port, "/health"
                    )
                    if status.healthy:
                        status.ready, status.last_error = _probe_endpoint(
                            self.args, status.port, "/v1/models"
                        )

                statuses.append(status)

            self.state.update_children(statuses)
            if self._failed_process is not None:
                break
            if self._stop_requested.wait(HEALTHCHECK_INTERVAL_S):
                break

    def _shutdown_children(self) -> None:
        if self.processes:
            logger.info(
                "Forwarding SIGTERM to %d local external LB child processes",
                len(self.processes),
            )
            shutdown_processes(self.processes, timeout=self.args.shutdown_timeout)
            self.processes = []

    def _shutdown_admin_server(self) -> None:
        if self._admin_server is not None:
            self._admin_server.shutdown()
            self._admin_server.server_close()
        if self._admin_thread is not None:
            self._admin_thread.join(timeout=5.0)
        self._admin_server = None
        self._admin_thread = None


def run_local_external_lb_supervisor(args: argparse.Namespace) -> None:
    LocalExternalLBSupervisor(args).run()
