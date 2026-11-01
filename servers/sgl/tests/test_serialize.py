from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from zonos3_sgl.message import BatchTTSBackendMsg, TTSSamplingParams, TTSUserMsg
from zonos3_sgl.message.utils import deserialize_type, serialize_type
from zonos3_sgl.utils.misc import call_if_main
from zonos3_sgl.utils.logger import init_logger

logger = init_logger(__name__)


@dataclass
class A:
    x: int
    y: str
    z: List[A]
    w: torch.Tensor


@call_if_main()
def test_serialize_deserialize():

    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    x = A(10, "hello", [A(20, "world", [], t)], t)
    data = serialize_type(x)
    logger.info(data)
    y = deserialize_type({"A": A}, data)
    logger.info(y)

    u = BatchTTSBackendMsg(
        [TTSUserMsg(uid=0, input_ids=t.view(1, -1), sampling_params=TTSSamplingParams())]
    )
    result = u.decoder(u.encoder())
    logger.info(u)
    logger.info(result)
