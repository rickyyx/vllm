# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.

import gc
import random
import time
from multiprocessing.managers import SharedMemoryManager

import msgspec
import pytest
import ray
import torch.distributed

from vllm.anyscale.shm.msgspec_shm import (RayEvent, SharedMsgspecBuffer,
                                           SharedMsgspecBufferWithEvent)
from vllm.anyscale.shm.msgspec_test import MockStruct


@ray.remote
class Actor:

    def __init__(self, input_buffer: SharedMsgspecBuffer,
                 output_buffer: SharedMsgspecBuffer):
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer

    def get_participant_id(self):
        return self.output_buffer.participant_id

    def run(self):
        try:
            while True:
                self.input_buffer.wait_for_incoming_data()
                data = self.input_buffer.get_data()
                self.input_buffer.clear()
                assert isinstance(data, MockStruct)
                time.sleep(random.uniform(0, 0.05))
                self.output_buffer.set_data(data)
        except Exception:
            self.output_buffer.set_error()
            raise

    def run_with_error(self):
        try:
            for i in range(4):
                self.input_buffer.wait_for_incoming_data()
                data = self.input_buffer.get_data()
                self.input_buffer.clear()
                assert isinstance(data, MockStruct)
                time.sleep(random.uniform(0, 0.05))
                self.output_buffer.set_data(data)
            raise ValueError()
        except Exception:
            self.output_buffer.set_error()
            raise

    def run_timeout(self):
        try:
            while True:
                self.input_buffer.wait_for_incoming_data()
                # will never finish
                time.sleep(10000000)
        except Exception:
            self.output_buffer.set_error()
            raise


@ray.remote(num_cpus=0)
class MultiActor:

    def __init__(self, input_buffer: SharedMsgspecBuffer,
                 output_buffer: SharedMsgspecBuffer, rank: int,
                 world_size: int):
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.rank = rank
        self.world_size = world_size

    def get_participant_id(self):
        return self.output_buffer.participant_id

    def init_torch_distributed(self, file_name: str):
        if torch.distributed.is_initialized():
            return
        torch.distributed.init_process_group(
            backend="gloo",
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{file_name}/file",
        )

    def run(self, participant_id):
        try:
            while True:
                self.input_buffer.wait_for_data_from(participant_id)
                # Explicitly synchronize.
                # Without this, we expect an error.
                data = self.input_buffer.get_data()
                torch.distributed.barrier()
                self.input_buffer.clear()
                output = [None] * self.world_size
                # Make sure all ranks have the exact same data
                torch.distributed.all_gather_object(output, data)
                time.sleep(random.uniform(0, 0.05))
                assert all(d == data for d in output)
                assert isinstance(data, MockStruct)
                if self.rank < 1:
                    self.output_buffer.set_data(data)
        except Exception:
            self.output_buffer.set_error()
            raise

    # Expected to fail
    def run_no_barrier(self, participant_id):
        try:
            while True:
                self.input_buffer.wait_for_data_from(participant_id)
                data = self.input_buffer.get_data()
                self.input_buffer.clear()
                output = [None] * self.world_size
                # Make sure all ranks have the exact same data
                torch.distributed.all_gather_object(output, data)
                time.sleep(random.uniform(0, 0.05))
                assert all(d == data for d in output)
                assert isinstance(data, MockStruct)
                if self.rank < 1:
                    self.output_buffer.set_data(data)
        except Exception:
            self.output_buffer.set_error()
            raise

    def run_with_error(self, participant_id):
        try:
            for i in range(4):
                self.input_buffer.wait_for_data_from(participant_id)
                data = self.input_buffer.get_data()
                torch.distributed.barrier()
                self.input_buffer.clear()
                assert isinstance(data, MockStruct)
                output = [None] * self.world_size
                # Make sure all ranks have the exact same data
                torch.distributed.all_gather_object(output, data)
                time.sleep(random.uniform(0, 0.05))
                assert all(d == data for d in output)
                if self.rank < 1:
                    self.output_buffer.set_data(data)
            raise ValueError()
        except Exception:
            self.output_buffer.set_error()
            raise


@pytest.fixture(autouse=True)
def cleanup():
    ray.shutdown()
    gc.collect()


@pytest.fixture
def manager():
    with SharedMemoryManager() as manager:
        yield manager


def _create_basic(manager):
    shared_buffer_input = SharedMsgspecBuffer(
        5e2,
        manager,
        msgspec.msgpack.Encoder,
        lambda: msgspec.msgpack.Decoder(MockStruct),
    )
    shared_buffer_output = SharedMsgspecBuffer(
        5e2,
        manager,
        msgspec.msgpack.Encoder,
        lambda: msgspec.msgpack.Decoder(MockStruct),
    )
    actor = Actor.remote(shared_buffer_input, shared_buffer_output)
    remote_participant_id = ray.get(actor.get_participant_id.remote())
    # Should be different for every process
    assert remote_participant_id != shared_buffer_output.participant_id
    return shared_buffer_input, shared_buffer_output, actor


def test_basic(manager):
    """Test basic usage (A writes -> B reads -> B writes -> A reads)"""
    shared_buffer_input, shared_buffer_output, actor = _create_basic(manager)
    actor.run.remote()

    random.seed(0)
    for _ in range(100):
        data = MockStruct.generate_random()
        shared_buffer_input.set_data(data)
        shared_buffer_output.wait_for_incoming_data()
        result = shared_buffer_output.get_data()
        assert result == data
        shared_buffer_output.clear()
    time.sleep(random.uniform(0, 0.05))


def test_error(manager):
    """Test that errors are propagated and don't hang"""
    shared_buffer_input, shared_buffer_output, actor = _create_basic(manager)
    future = actor.run_with_error.remote()

    random.seed(0)
    with pytest.raises(RuntimeError):
        for _ in range(5):
            data = MockStruct.generate_random()
            shared_buffer_input.set_data(data)
            shared_buffer_output.wait_for_incoming_data()
            result = shared_buffer_output.get_data()
            assert result == data
            shared_buffer_output.clear()

    with pytest.raises((ValueError, RuntimeError)):
        ray.get(future)


def test_timeout(manager):
    """Test timeout works"""
    shared_buffer_input, shared_buffer_output, actor = _create_basic(manager)
    actor.run_timeout.remote()

    data = MockStruct.generate_random()
    shared_buffer_input.set_data(data)
    with pytest.raises(TimeoutError):
        shared_buffer_output.wait_for_incoming_data(timeout_s=1)


def _create_with_event(manager):
    event = RayEvent.remote()
    shared_buffer_input = SharedMsgspecBufferWithEvent(
        5e2,
        manager,
        msgspec.msgpack.Encoder,
        lambda: msgspec.msgpack.Decoder(MockStruct),
        event,
    )
    shared_buffer_output = SharedMsgspecBufferWithEvent(
        5e2,
        manager,
        msgspec.msgpack.Encoder,
        lambda: msgspec.msgpack.Decoder(MockStruct),
        event,
    )
    actor = Actor.remote(shared_buffer_input, shared_buffer_output)
    remote_participant_id = ray.get(actor.get_participant_id.remote())
    # Should be different for every process
    assert remote_participant_id != shared_buffer_output.participant_id
    return shared_buffer_input, shared_buffer_output, actor


def test_basic_with_event(manager):
    """Test basic usage (A writes -> B reads -> B writes -> A reads)"""
    shared_buffer_input, shared_buffer_output, actor = _create_with_event(
        manager)
    actor.run.remote()

    random.seed(0)
    for _ in range(10):
        shared_buffer_input.wake_up()
        shared_buffer_output.wake_up()
        for _ in range(10):
            data = MockStruct.generate_random()
            shared_buffer_input.set_data(data)
            shared_buffer_output.wait_for_incoming_data()
            result = shared_buffer_output.get_data()
            assert result == data
            shared_buffer_output.clear()
        shared_buffer_input.put_to_sleep()
        shared_buffer_output.put_to_sleep()
        time.sleep(random.uniform(0, 0.05))


def test_with_event_error(manager):
    """Test that errors are propagated and don't hang"""
    shared_buffer_input, shared_buffer_output, actor = _create_with_event(
        manager)
    future = actor.run_with_error.remote()

    random.seed(0)
    with pytest.raises(RuntimeError):
        for _ in range(5):
            data = MockStruct.generate_random()
            shared_buffer_input.set_data(data)
            shared_buffer_output.wait_for_incoming_data()
            result = shared_buffer_output.get_data()
            assert result == data
            shared_buffer_output.clear()

    with pytest.raises((ValueError, RuntimeError)):
        ray.get(future)


def test_with_event_timeout(manager):
    """Test timeout works"""
    shared_buffer_input, shared_buffer_output, actor = _create_with_event(
        manager)
    actor.run_timeout.remote()

    data = MockStruct.generate_random()
    shared_buffer_input.set_data(data)
    with pytest.raises(TimeoutError):
        shared_buffer_output.wait_for_incoming_data(timeout_s=1)


def _create_distributed(manager, tmpdir):
    event = RayEvent.remote()
    shared_buffer_input = SharedMsgspecBufferWithEvent(
        5e2,
        manager,
        msgspec.msgpack.Encoder,
        lambda: msgspec.msgpack.Decoder(MockStruct),
        event,
    )
    shared_buffer_output = SharedMsgspecBufferWithEvent(
        5e2,
        manager,
        msgspec.msgpack.Encoder,
        lambda: msgspec.msgpack.Decoder(MockStruct),
        event,
    )
    actors = [
        MultiActor.remote(shared_buffer_input, shared_buffer_output, i, 8)
        for i in range(8)
    ]
    ray.get(
        [actor.init_torch_distributed.remote(str(tmpdir)) for actor in actors])
    participant_ids = ray.get(
        [actor.get_participant_id.remote()
         for actor in actors]) + [shared_buffer_output.participant_id]
    # Should be different for every process
    assert len(set(participant_ids)) == len(participant_ids)
    return shared_buffer_input, shared_buffer_output, actors


def test_distributed_with_event(manager, tmpdir):
    """Test basic usage with multiple readers (worker group)"""
    shared_buffer_input, shared_buffer_output, actors = _create_distributed(
        manager, tmpdir)
    futures = [
        actor.run.remote(shared_buffer_input.participant_id)
        for actor in actors
    ]
    print(futures)

    random.seed(0)
    for _ in range(5):
        for _ in range(200):
            data = MockStruct.generate_random()
            shared_buffer_input.set_data(data)
            shared_buffer_output.wait_for_incoming_data()
            result = shared_buffer_output.get_data()
            assert result == data
            shared_buffer_output.clear()
        shared_buffer_input.put_to_sleep()
        shared_buffer_output.put_to_sleep()
        time.sleep(random.uniform(0, 0.05))


@pytest.mark.skip("Test is flaky (data sometimes doesn't diverge)")
def test_distributed_with_event_no_barrier(manager, tmpdir):
    """Test that lack of explicit synchronization in workers
    causes data to diverge and thus an error."""
    shared_buffer_input, shared_buffer_output, actors = _create_distributed(
        manager, tmpdir)
    futures = [
        actor.run_no_barrier.remote(shared_buffer_input.participant_id)
        for actor in actors
    ]
    print(futures)

    random.seed(0)
    with pytest.raises(RuntimeError):
        for _ in range(5):
            for _ in range(200):
                data = MockStruct.generate_random()
                shared_buffer_input.set_data(data)
                shared_buffer_output.wait_for_incoming_data()
                result = shared_buffer_output.get_data()
                assert result == data
                shared_buffer_output.clear()
            shared_buffer_input.put_to_sleep()
            shared_buffer_output.put_to_sleep()
            time.sleep(random.uniform(0, 0.05))


def test_distributed_with_event_with_error(manager, tmpdir):
    """Test that errors are propagated and don't hang"""
    shared_buffer_input, shared_buffer_output, actors = _create_distributed(
        manager, tmpdir)
    futures = [
        actor.run_with_error.remote(shared_buffer_input.participant_id)
        for actor in actors
    ]

    random.seed(0)
    with pytest.raises(RuntimeError):
        for _ in range(5):
            data = MockStruct.generate_random()
            shared_buffer_input.set_data(data)
            shared_buffer_output.wait_for_incoming_data()
            result = shared_buffer_output.get_data()
            assert result == data
            shared_buffer_output.clear()

    with pytest.raises((ValueError, RuntimeError)):
        ray.get(futures)
