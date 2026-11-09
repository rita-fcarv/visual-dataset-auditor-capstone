import os
import random

OUTPUT = "data/validation"
WIDTH = 128
HEIGHT = 128

os.makedirs(OUTPUT, exist_ok=True)


def save_ppm(filename, pixels):
    with open(os.path.join(OUTPUT, filename), "wb") as file:
        file.write(
            f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii")
        )
        file.write(bytes(pixels))


random.seed(42)

# Base image with reproducible spatial patterns.
base = bytearray()

for y in range(HEIGHT):
    for x in range(WIDTH):
        value = (3 * x + 5 * y + (x * y) % 97) % 256
        base.extend((
            value,
            (value + x) % 256,
            (value + y) % 256,
        ))

save_ppm("original.ppm", base)

# Exact duplicate.
save_ppm("exact_copy.ppm", base)

# Slight brightness modification.
brighter = bytearray(
    min(255, value + 10) for value in base
)
save_ppm("brighter.ppm", brighter)

# Unrelated random image.
different = bytearray(
    random.randrange(256)
    for _ in range(WIDTH * HEIGHT * 3)
)
save_ppm("different.ppm", different)

print("Generated 4 validation images.")
