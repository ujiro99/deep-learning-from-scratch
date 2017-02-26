from typing import Union

num = Union[int, float]


class MulLayer:

    def __init__(self):
        self.x = None  # type: num
        self.y = None  # type: num

    def forward(self, x: num, y: num) -> num:
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout: num) -> num:
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
