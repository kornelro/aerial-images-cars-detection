from abc import ABC, abstractmethod
from typing import Generator, Set, Tuple

import numpy as np


class Slider(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self,
        image: np.array
    ) -> Generator[Set[float], None, None]:
        pass


class SlidingWindow(Slider):

    def __init__(
        self,
        step_size: int,
        window_size: Tuple[int, int]
    ):
        super().__init__()
        self.step_size = step_size
        self.window_size = window_size

    def __call__(
        self,
        image: np.array
    ) -> Generator[Set[float], None, None]:

        for bottom_left_x in range(0, image.shape[1], self.step_size):
            for bottom_left_y in range(0, image.shape[0], self.step_size):

                top_left_x = bottom_left_x
                top_left_y = bottom_left_y + self.window_size[1]

                top_right_x = top_left_x + self.window_size[0]
                top_right_y = top_left_y

                bottom_right_x = bottom_left_x + self.window_size[0]
                bottom_right_y = bottom_left_y

                yield (
                    top_left_x, top_left_y,
                    top_right_x, top_right_y,
                    bottom_left_x, bottom_left_y,
                    bottom_right_x, bottom_right_y
                )
