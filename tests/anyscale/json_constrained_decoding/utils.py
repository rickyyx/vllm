import random
import time
from typing import List, Tuple

from vllm.anyscale.constrained_decoding.logits_processor import (
    JSONModeLogitsProcessor)


# For testing purposes
class FaultyJsonModeLogitsProcessor(JSONModeLogitsProcessor):
    """Logit Processor with two type of faults:
        1. Taking too long (This will cause the shared memory to timeout with
            no error / data)
        2. Some error gets raised in the processor (This will cause the shared
            memory to have an error)
    """

    def __init__(
        self,
        *args,
        p_timeout: float = 0.01,
        p_error: float = 0.01,
        **kwargs,
    ):
        self._p_timeout = p_timeout
        self._p_error = p_error
        super().__init__(*args, **kwargs)

    def call(self, input_list: List[Tuple[List[int], str]]):
        rand_n = random.random()
        if rand_n < self._p_timeout:
            print(f"Faulty token ids {rand_n=} (taking too long ...)")
            time.sleep(50)

        rand_n = random.random()
        if rand_n < self._p_error:
            raise ValueError(f"Faulty token ids {rand_n=}")

        return super().call(input_list)


class FaultyProcessorWithException(FaultyJsonModeLogitsProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, p_timeout=0.0, p_error=0.01, **kwargs)


class FaultyProcessorWithErrorsAllTheTime(FaultyJsonModeLogitsProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, p_timeout=0.0, p_error=1.0, **kwargs)


class ProcessorFailFirstRank(JSONModeLogitsProcessor):
    """Logit Processor that only fails when it is the first rank.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def call(self, input_list: List[Tuple[List[int], str]]):
        if self._rank == 0:
            raise ValueError("Faulty token ids")

        return super().call(input_list)
