# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.
"""
Test utils. Due to a msgspec serialization quirk, they have to be
defined here and not in tests. (https://github.com/jcrist/msgspec/issues/394)
"""

import random
from typing import Dict, List

import msgspec


class MockStructInner(msgspec.Struct, array_like=True):
    a: int
    b: float

    @classmethod
    def generate_random(cls):
        return cls(
            a=random.randint(0, 100),
            b=random.random(),
        )


class MockStruct(msgspec.Struct, array_like=True):
    a: int
    b: float
    c: str
    d: List[str]
    e: Dict[int, List[int]]
    f: List[MockStructInner]

    @classmethod
    def generate_random(cls):
        return cls(
            a=random.randint(0, 100),
            b=random.random(),
            c=str(random.randint(0, 100)),
            d=[
                str(random.randint(0, 100))
                for _ in range(random.randint(0, 10))
            ],
            e={
                i:
                [random.randint(0, 100) for _ in range(random.randint(0, 10))]
                for i in range(random.randint(0, 10))
            },
            f=[
                MockStructInner.generate_random()
                for _ in range(random.randint(0, 10))
            ],
        )
