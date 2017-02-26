from typing import Union

num = Union[int, float]


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x: num, y: num) -> num:
        return x + y

    def backward(self, dout: num) -> num:
        dx = dout * 1
        dy = dout * 1
        return dx, dy
