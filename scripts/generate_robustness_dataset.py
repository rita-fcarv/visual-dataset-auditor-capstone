import argparse
import os
import random

WIDTH = 128
HEIGHT = 128


def save_ppm(path, pixels):
    with open(path, "wb") as file:
        file.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        file.write(bytes(pixels))


def create_image(seed):
    rng = random.Random(seed)
    pixels = bytearray(WIDTH * HEIGHT * 3)

    # Spatially varying gradients, stripes and rectangular regions.
    ax = rng.randint(1, 7)
    ay = rng.randint(1, 7)
    phase = rng.randint(0, 255)
    stripe_period = rng.randint(8, 28)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            index = (y * WIDTH + x) * 3

            gradient = (ax * x + ay * y + phase) % 256
            stripe = 50 if (x // stripe_period) % 2 == 0 else -30
            region = 45 if (x - 64) ** 2 + (y - 64) ** 2 < 1200 else 0

            value = max(0, min(255, gradient + stripe + region))

            pixels[index] = value
            pixels[index + 1] = max(0, min(255, value + (x % 37) - 18))
            pixels[index + 2] = max(0, min(255, value + (y % 43) - 21))

    return pixels


def brighten(pixels):
    return bytearray(min(255, value + 20) for value in pixels)


def blur_image(pixels):
    # 3x3 box blur, implemented without external dependencies.
    output = bytearray(len(pixels))

    for y in range(HEIGHT):
        for x in range(WIDTH):
            for channel in range(3):
                total = 0
                count = 0

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nx = min(WIDTH - 1, max(0, x + dx))
                        ny = min(HEIGHT - 1, max(0, y + dy))

                        index = (ny * WIDTH + nx) * 3 + channel
                        total += pixels[index]
                        count += 1

                output[(y * WIDTH + x) * 3 + channel] = total // count

    return output


def resize_roundtrip(pixels):
    # Downscale 128x128 to 64x64 using 2x2 averaging,
    # then upscale using nearest-neighbour interpolation.
    output = bytearray(len(pixels))

    for y in range(HEIGHT):
        for x in range(WIDTH):
            source_x = (x // 2) * 2
            source_y = (y // 2) * 2

            for channel in range(3):
                total = 0

                for dy in (0, 1):
                    for dx in (0, 1):
                        index = (
                            ((source_y + dy) * WIDTH + source_x + dx)
                            * 3 + channel
                        )
                        total += pixels[index]

                output[(y * WIDTH + x) * 3 + channel] = total // 4

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", default="data/robustness")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    os.makedirs(args.output, exist_ok=True)

    # Clear previously generated PPMs to avoid stale test images.
    for filename in os.listdir(args.output):
        if filename.endswith(".ppm"):
            os.remove(os.path.join(args.output, filename))

    for index in range(args.count):
        prefix = f"group_{index:03d}"

        original = create_image(1000 + index)
        unrelated = create_image(100000 + index)

        variants = {
            "original": original,
            "exact": original,
            "bright": brighten(original),
            "blur": blur_image(original),
            "resized": resize_roundtrip(original),
            "different": unrelated,
        }

        for variant, pixels in variants.items():
            filename = f"{prefix}_{variant}.ppm"
            save_ppm(os.path.join(args.output, filename), pixels)

    print(f"Generated {args.count * 6} images.")
    print(f"Groups: {args.count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
