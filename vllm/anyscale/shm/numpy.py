import msgspec.msgpack
import numpy as np


class NumpyStruct(msgspec.Struct, array_like=True):
    dtype: str
    shape: tuple
    data: bytes


numpy_array_encoder = msgspec.msgpack.Encoder()
numpy_array_decoder = msgspec.msgpack.Decoder(type=NumpyStruct)


def numpy_encode_hook(obj):
    if isinstance(obj, np.ndarray):
        return msgspec.msgpack.Ext(
            1,
            numpy_array_encoder.encode(
                NumpyStruct(dtype=obj.dtype.str,
                            shape=obj.shape,
                            data=obj.data)))
    return obj


def numpy_ext_hook(msg_type, data: memoryview):
    if msg_type == 1:
        serialized_array_rep = numpy_array_decoder.decode(data)
        return np.frombuffer(serialized_array_rep.data,
                             dtype=serialized_array_rep.dtype).reshape(
                                 serialized_array_rep.shape)
    return data
