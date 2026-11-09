import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", default="data/images")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be positive")

    os.makedirs(args.output, exist_ok=True)

    width = 128
    height = 128
    random.seed(42)

    for index in range(args.count):
        path = os.path.join(
            args.output, f"image_{index:05d}.ppm"
        )

        red = random.randrange(256)
        green = random.randrange(256)
        blue = random.randrange(256)

        with open(path, "wb") as image:
            image.write(
                f"P6\n{width} {height}\n255\n".encode("ascii")
            )

            for y in range(height):
                for x in range(width):
                    pixel = bytes((
                        (red + x) % 256,
                        (green + y) % 256,
                        (blue + x + y) % 256,
                    ))
                    image.write(pixel)

    print(f"Generated {args.count} images in {args.output}")


if __name__ == "__main__":
    main()
