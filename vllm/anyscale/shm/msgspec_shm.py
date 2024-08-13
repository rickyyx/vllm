# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.

import asyncio
import logging
import secrets
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Callable, Optional

import msgspec
import numpy as np
import ray

SLEEP_TIME = 0.0001
ERROR_CODE = 255
logger = logging.getLogger(__name__)

RayEvent = ray.remote(num_cpus=0)(asyncio.Event)


class SharedMemoryError(RuntimeError):
    pass


class SharedMemoryWriteDataError(SharedMemoryError):
    pass


class SharedMemoryReadDataError(SharedMemoryError):
    pass


class SharedMsgspecBuffer:
    """Simple shared memory buffer for bi-directional communication.

    The buffer is implemented as a shared memory array of bytes. The data
    is encoded by msgspec.msgpack.

    The buffer is intended to be used by two processes exchanging messages
    (A writes -> B reads -> B writes -> A reads). There are NO synchronization
    guarantees. The buffer can support multiple readers but only if external
    synchronization (eg. locks, torch.distributed.barrier) are used.
    """

    def __init__(
        self,
        size: int,
        manager: SharedMemoryManager,
        encoder_init_fn: Callable[[], msgspec.msgpack.Encoder],
        decoder_init_fn: Callable[[], msgspec.msgpack.Decoder],
    ) -> None:
        if size >= np.iinfo(np.uint64).max:
            raise ValueError(
                f"Size must be less than {np.iinfo(np.uint64).max}, got {size}."
            )
        size = int(size)
        self.encoder_init_fn = encoder_init_fn
        self.decoder_init_fn = decoder_init_fn
        self.encoder = encoder_init_fn()
        self.decoder = decoder_init_fn()
        self.participant_id = self._generate_id()
        meta_array = self._create_meta_array()
        self._data_buffer = manager.SharedMemory(size + meta_array.nbytes)
        self._offset = meta_array.nbytes
        self._meta_array = np.ndarray(
            shape=meta_array.shape,
            dtype=meta_array.dtype,
            buffer=self._data_buffer.buf[:self._offset],
        )
        self.clear()

    def _create_meta_array(self) -> np.ndarray:
        # size, buffer id, extra metadata
        meta_array = np.array([0, self.participant_id, 0], dtype=np.uint64)
        return meta_array

    @property
    def _data_size(self) -> int:
        return self._meta_array[0]

    @_data_size.setter
    def _data_size(self, value: int):
        self._meta_array[0] = value

    @property
    def _writer_id(self) -> int:
        return self._meta_array[1]

    @_writer_id.setter
    def _writer_id(self, value: int):
        self._meta_array[1] = value

    @property
    def _meta(self) -> int:
        return self._meta_array[2]

    @_meta.setter
    def _meta(self, value: int):
        self._meta_array[2] = value

    def _generate_id(self) -> int:
        return secrets.randbits(64)

    def has_data(self) -> bool:
        """Returns True if the buffer has ANY data."""
        return self.has_errored() or self._data_size > 0

    def has_incoming_data(self) -> bool:
        """Returns True if the buffer has data from a different process."""
        return self.has_errored() or (self._data_size > 0 and
                                      self._writer_id != self.participant_id)

    def has_data_from(self, participant_id: int):
        """Returns True if the buffer has data from the specified process."""
        return self.has_errored() or (self._data_size > 0
                                      and self._writer_id == participant_id)

    def has_errored(self):
        """Returns True if there has been an error."""
        return self._meta >= ERROR_CODE

    def wait_for_data_from(self, participant_id: int):
        """Blocks until there is data from the specified process.

        This uses a busy loop."""
        while not self.has_data_from(participant_id):
            time.sleep(SLEEP_TIME)

    async def wait_for_data_from_async(self, participant_id: int):
        """Blocks until there is data from the specified process.

        This uses a busy loop."""
        while not self.has_data_from(participant_id):
            await asyncio.sleep(SLEEP_TIME)
            # Threading yield
            time.sleep(0)

    def wait_for_incoming_data(self, *, timeout_s: Optional[float] = None):
        """Blocks until there is data from another process.

        This uses a busy loop."""
        if timeout_s is None:
            return self._wait_for_incoming_data()
        return self._wait_for_incoming_data_timeout(timeout_s)

    def _wait_for_incoming_data(self):
        while not self.has_incoming_data():
            time.sleep(SLEEP_TIME)

    def _wait_for_incoming_data_timeout(self, timeout_s: float):
        st = time.monotonic()
        while not self.has_incoming_data():
            if time.monotonic() - st > timeout_s:
                raise TimeoutError(
                    f"Timed out while waiting {timeout_s}s for incoming data.")
            time.sleep(SLEEP_TIME)

    async def wait_for_incoming_data_async(self):
        """Blocks until there is data from another process.

        This uses a busy loop."""
        while not self.has_incoming_data():
            await asyncio.sleep(SLEEP_TIME)
            # Threading yield
            time.sleep(0)

    def clear(self):
        self._data_size = 0

    def clear_error(self):
        self._meta = 0

    def set_error(self):
        self._meta = ERROR_CODE

    def set_data(self, data: msgspec.Struct) -> None:
        """Encodes and writes msgspec data into buffer."""
        self.clear()
        data = self.encoder.encode(data)
        data_len = len(data)
        self._data_buffer.buf[self._offset:data_len + self._offset] = data
        if self.has_errored():
            raise SharedMemoryWriteDataError("Error in remote buffer")
        self._writer_id = self.participant_id
        self._data_size = data_len

    def get_data(self) -> Optional[msgspec.Struct]:
        """Reads data from the buffer. Returns None if there is no data."""
        if self.has_errored():
            raise SharedMemoryReadDataError("Error in remote buffer")
        size = int(self._data_size)
        if size <= 0:
            return None
        ret = self.decoder.decode(
            self._data_buffer.buf[self._offset:self._offset + size])
        return ret

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_incoming_participant_id"] = state.pop("participant_id")
        state.pop("encoder")
        state.pop("decoder")
        state.pop("_meta_array")
        return state

    def __setstate__(self, state: dict):
        incoming_participant_id = state.pop("_incoming_participant_id")
        self.__dict__.update(state)
        for _ in range(10):
            self.participant_id = self._generate_id()
            if self.participant_id != incoming_participant_id:
                break
            logger.warning(
                "Generated same participant id (%s vs old "
                "%s), retrying.", self.participant_id, incoming_participant_id)
        else:
            raise RuntimeError(
                "Failed to generate unique participant id after 10 tries.")
        meta_array = self._create_meta_array()
        self._offset = meta_array.nbytes
        self._meta_array = np.ndarray(
            shape=meta_array.shape,
            dtype=meta_array.dtype,
            buffer=self._data_buffer.buf[:self._offset],
        )
        self.encoder = self.encoder_init_fn()
        self.decoder = self.decoder_init_fn()


class SharedMsgspecBufferWithEvent(SharedMsgspecBuffer):
    """Simple shared memory buffer for bi-directional communication.

    The buffer is implemented as a shared memory array of bytes. The data
    is encoded by msgspec.msgpack.

    This buffer avoids having to busy loop constantly by using a Ray
    Event ("sleeping).

    Since awaiting for a Ray Event is slower than busy looping, it is possible
    to switch between the two modes. An example would be to busy loop when
    you are in the middle of generating tokens in a batch, but switching
    to event when awaiting a new batch.

    The buffer is intended to be used by two processes exchanging messages
    (A writes -> B reads -> B writes -> A reads). There are NO synchronization
    guarantees. The buffer can support multiple readers
    (A writes -> B & C & D read -> B writes -> A reads) but only if external
    synchronization (eg. locks, torch.distributed.barrier) are used.
    """

    def __init__(
        self,
        size: int,
        manager: SharedMemoryManager,
        encoder_init_fn: Callable[[], msgspec.msgpack.Encoder],
        decoder_init_fn: Callable[[], msgspec.msgpack.Decoder],
        ray_event: RayEvent,
    ) -> None:
        super().__init__(
            size=size,
            manager=manager,
            encoder_init_fn=encoder_init_fn,
            decoder_init_fn=decoder_init_fn,
        )
        self.ray_event = ray_event
        self.clear()
        self.put_to_sleep(block=True)

    @property
    def sleeping(self) -> bool:
        """Returns True if the buffer is sleeping."""
        return self._meta == 1

    def put_to_sleep(self, block: bool = True):
        """Puts the buffer to sleep."""
        if self.has_data():
            raise RuntimeError("Cannot put buffer to sleep when not empty.")
        fut = self.ray_event.clear.remote()
        if block:
            ray.get(fut)
        self._meta = 1
        return fut

    def set_error(self):
        super().set_error()
        self.wake_up(block=True)

    def wake_up(self, set_event: bool = True, block: bool = False):
        """Wakes up the buffer."""
        fut = None
        if set_event:
            fut = self.ray_event.set.remote()
            if block:
                ray.get(fut)
        if self._meta == 1:
            self._meta = 0
        return fut

    def set_data(self, data: msgspec.Struct, wake_up: bool = True) -> None:
        super().set_data(data)
        if wake_up:
            self.wake_up()

    def _wait_for_incoming_data(self):
        while not self.has_incoming_data():
            if self.sleeping:
                ray.get(self.ray_event.wait.remote())
                self.wake_up(set_event=False)
            else:
                time.sleep(SLEEP_TIME)

    def _wait_for_incoming_data_timeout(self, timeout_s: float):
        st = time.monotonic()
        while not self.has_incoming_data():
            if self.sleeping:
                ray.get(self.ray_event.wait.remote(), timeout=timeout_s)
                self.wake_up(set_event=False)
            if time.monotonic() - st > timeout_s:
                raise TimeoutError(
                    f"Timed out while waiting {timeout_s}s for incoming data.")
            time.sleep(SLEEP_TIME)

    async def wait_for_incoming_data_async(self):
        while not self.has_incoming_data():
            if self.sleeping:
                await self.ray_event.wait.remote()
                self.wake_up(set_event=False)
            else:
                await asyncio.sleep(SLEEP_TIME)
                # Threading yield
                time.sleep(0)

    def wait_for_data_from(self, participant_id: int):
        while not self.has_data_from(participant_id):
            if self.sleeping:
                ray.get(self.ray_event.wait.remote())
                self.wake_up(set_event=False)
            else:
                time.sleep(SLEEP_TIME)

    async def wait_for_data_from_async(self, participant_id: int):
        while not self.has_data_from(participant_id):
            if self.sleeping:
                await self.ray_event.wait.remote()
                self.wake_up(set_event=False)
            else:
                await asyncio.sleep(SLEEP_TIME)
                # Threading yield
                time.sleep(0)
