from collections import namedtuple
from dataclasses import dataclass
from typing import TypedDict, NewType, Tuple, Any

import numpy as np

from .graphics import Texture

TileKind = NewType("TileKind", str)


class TileDict(TypedDict):
    # {"coords": (i, j), "kind": kind, "angle": angle, "drivable": drivable})
    coords: Tuple[int, int]
    kind: TileKind
    angle: int
    drivable: bool
    texture: Texture
    color: np.ndarray
    curves: Any


@dataclass
class DoneRewardInfo:
    done: bool
    done_why: str
    done_code: str
    reward: float


@dataclass
class DynamicsInfo:
    motor_left: float
    motor_right: float


LanePosition0 = namedtuple("LanePosition", "dist dot_dir angle_deg angle_rad")


class LanePosition(LanePosition0):
    def as_json_dict(self):
        """Serialization-friendly format."""
        return dict(
            dist=self.dist,
            dot_dir=self.dot_dir,
            angle_deg=self.angle_deg,
            angle_rad=self.angle_rad,
        )
