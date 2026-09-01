# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Remote synchronous inference engine: chunk-at-a-time policy calls to a
``lerobot.async_inference.policy_server`` running on another machine.

Fork addition (2026-09). Purpose: run the standard ``lerobot-rollout``
protocol (episodic strategy, recording, end-episode key, warm-up convention)
with inference served from a cloud GPU. Semantics are deliberately *sync*:
request a chunk, block until it arrives, execute every action in it, request
the next. No overlap, no aggregation, no staleness — the only variable versus
``SyncInferenceEngine`` is dead time per chunk (~0.34s remote A10 vs ~1s MPS,
measured; see perturbation/follow-up results/running_inference_on_cloud.md).

Design notes:
- ``get_action`` receives the dataset-style ``obs_frame`` the rollout loop
  builds per tick (``observation.state`` vector, ``observation.images.<cam>``
  HWC uint8). The policy server wants the *raw robot observation* (per-motor
  keys + camera names) because it rebuilds the frame itself, so we invert
  ``build_dataset_frame`` here using the feature ``names``.
- Camera rename (``front`` -> ``camera1``) is applied on the wire: the server's
  ``prepare_raw_observation`` indexes ``policy_image_features`` by the incoming
  key, with no rename layer. The checkpoint preprocessor (server-side) also
  tolerates the missing ``front`` key.
- The server applies the checkpoint's pre/post-processors; returned actions
  are already unnormalized. We only reorder to ``ordered_action_keys`` for
  parity with the sync engine.
- The locally loaded ``policy`` / processors passed by the factory are unused
  (the rollout entry point loads them regardless); inference lives on the box.
- Requires the server's ``helpers.resize_robot_observation_image`` pass-through
  patch (v2); the upstream squash causes a lateral grasp offset.
"""

from __future__ import annotations

import logging
import pickle  # nosec
import time
from collections import deque

import grpc
import numpy as np
import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.policies.utils import make_robot_action
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .base import InferenceEngine

logger = logging.getLogger(__name__)

OBS_PREFIX = "observation."
IMG_PREFIX = "observation.images."


class RemoteSyncInferenceEngine(InferenceEngine):
    """Sync semantics, remote compute. One blocking chunk request per queue drain."""

    def __init__(
        self,
        *,
        server_address: str,
        policy_type: str,
        policy_path_on_server: str,
        policy_device: str,
        actions_per_chunk: int,
        image_rename: dict[str, str],
        chunk_timeout_s: float,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        fps: float,
    ) -> None:
        if not policy_path_on_server:
            raise ValueError("remote inference needs --inference.policy_path_on_server (path valid ON THE SERVER)")
        self._server_address = server_address
        self._policy_type = policy_type
        self._policy_path = policy_path_on_server
        self._policy_device = policy_device
        self._actions_per_chunk = actions_per_chunk
        self._image_rename = image_rename
        self._chunk_timeout_s = chunk_timeout_s
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._dt = 1.0 / fps

        # Observation features as the server expects them (dataset-style dict
        # with dtype/shape/names), with the camera rename applied.
        self._lerobot_features = {
            self._image_rename.get(k, k): v
            for k, v in dataset_features.items()
            if k.startswith(OBS_PREFIX)
        }
        self._state_names = list(dataset_features["observation.state"]["names"])
        self._image_keys = [k for k in dataset_features if k.startswith(IMG_PREFIX)]

        self._channel = None
        self._stub = None
        self._fifo: deque[torch.Tensor] = deque()
        self._timestep = 0
        self._chunk_count = 0
        self._latency_log: list[float] = []

        logger.info(
            "RemoteSyncInferenceEngine initialized (server=%s, policy=%s@%s on %s, chunk=%d)",
            server_address, policy_type, policy_path_on_server, policy_device, actions_per_chunk,
        )

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> None:
        self._channel = grpc.insecure_channel(
            self._server_address, grpc_channel_options(initial_backoff=f"{self._dt:.4f}s")
        )
        self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)
        t0 = time.perf_counter()
        self._stub.Ready(services_pb2.Empty())
        logger.info("Connected to policy server in %.3fs", time.perf_counter() - t0)

        policy_cfg = RemotePolicyConfig(
            self._policy_type,
            self._policy_path,
            self._lerobot_features,
            self._actions_per_chunk,
            self._policy_device,
        )
        self._stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_cfg)))
        logger.info("RemoteSyncInferenceEngine started (server loading %s)", self._policy_path)

    def stop(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        if self._latency_log:
            med = sorted(self._latency_log)[len(self._latency_log) // 2]
            logger.info(
                "RemoteSyncInferenceEngine stopped (%d chunks, median round trip %.3fs)",
                self._chunk_count, med,
            )

    def reset(self) -> None:
        """Episode boundary: drop any unexecuted actions, restart timestep."""
        logger.info("Resetting remote inference state (fifo=%d dropped)", len(self._fifo))
        self._fifo.clear()
        self._timestep = 0

    # ---- action production -------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None
        if not self._fifo:
            self._request_chunk(obs_frame)
        if not self._fifo:  # server returned nothing usable
            return None
        self._timestep += 1
        return self._fifo.popleft()

    # ---- internals -------------------------------------------------------

    def _to_raw_observation(self, obs_frame: dict) -> dict:
        """Invert build_dataset_frame: dataset-style frame -> raw robot obs dict."""
        raw: dict = {}
        state = obs_frame["observation.state"]
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
        for name, value in zip(self._state_names, state.tolist(), strict=True):
            raw[name] = float(value)
        for key in self._image_keys:
            img = obs_frame[key]
            img = img.cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
            wire_key = self._image_rename.get(key, key)
            raw[wire_key.removeprefix(IMG_PREFIX)] = img  # server re-prefixes via lerobot_features
        raw["task"] = self._task
        return raw

    def _request_chunk(self, obs_frame: dict) -> None:
        obs = TimedObservation(
            timestamp=time.time(),
            observation=self._to_raw_observation(obs_frame),
            timestep=self._timestep,
        )
        obs.must_go = True  # sync: every request is a fresh plan, never skipped

        t0 = time.perf_counter()
        self._stub.SendObservations(
            send_bytes_in_chunks(
                pickle.dumps(obs), services_pb2.Observation, log_prefix="[REMOTE] Observation", silent=True
            )
        )
        # Block until the server has a chunk for us.
        deadline = t0 + self._chunk_timeout_s
        while True:
            reply = self._stub.GetActions(services_pb2.Empty())
            if len(reply.data) > 0:
                break
            if time.perf_counter() > deadline:
                logger.error("Remote chunk request timed out after %.1fs", self._chunk_timeout_s)
                return
            time.sleep(0.005)
        round_trip = time.perf_counter() - t0
        self._latency_log.append(round_trip)
        self._chunk_count += 1

        timed_actions = pickle.loads(reply.data)  # nosec
        for ta in timed_actions:
            action_tensor = ta.get_action().detach().cpu()
            action_dict = make_robot_action(action_tensor, self._dataset_features)
            self._fifo.append(torch.tensor([action_dict[k] for k in self._ordered_action_keys]))

        logger.debug(
            "Chunk #%d: %d actions, round trip %.3fs", self._chunk_count, len(timed_actions), round_trip
        )
