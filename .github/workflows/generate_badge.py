# ruff: noqa: E471
import argparse
import colorsys
import re
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_TEMPLATE = """\
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="{61 + width//10 + 10}"
  height="20"
  role="img"
  aria-label="coverage: {coverage}"
>
    <title>coverage: {coverage}</title>
    <g shape-rendering="crispEdges">
        <rect width="61" height="20" fill="#555"/>
        <rect x="61" width="{width//10 + 10}" height="20" fill="{color}"/>
    </g>
    <g
      fill="#fff"
      text-anchor="left"
      font-family="Verdana,Geneva,DejaVu Sans,sans-serif"
      text-rendering="geometricPrecision"
      font-size="110"
    >
        <text x="60" y="140" transform="scale(.1)" fill="#fff" textLength="510">coverage</text>
        <text x="650" y="140" transform="scale(.1)" fill="#fff" textLength="{width}">{coverage}</text>
    </g>
</svg>"""  # noqa: E501


def parse_any_color_as_hsl(color: str) -> Tuple[float, float, float]:
    if color.startswith("hsl"):
        h, s, l = tuple(map(float, re.findall(r"\d+(?:\.\d+)?", color)))[:3]
        return (h / 360, s / 100, l / 100)
    elif color.startswith("#"):
        rgb = tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
        h, l, s = colorsys.rgb_to_hls(*rgb)
    elif color.startswith("rgb"):
        h, l, s = colorsys.rgb_to_hls(*map(int, re.findall(r"\d+", color)))
    else:
        raise ValueError(f"Unknown color format: {color}")
    return h, s, l


def make_badge(
    bad_color_hsl: Tuple[float, float, float],
    good_color_hsl: Tuple[float, float, float],
    min_coverage: int = 70,
    report_path: str = "coverage.txt",
    badge_template_path: Optional[str] = None,
):
    with open(report_path) as f:
        coverage = float(f.read().splitlines(False)[-1].split()[-1].strip(" %"))
    coverage_str = str(int(coverage)) + "%"
    ratio = (max(coverage, min_coverage) - min_coverage) / (100 - min_coverage)
    hue = bad_color_hsl[0] + ratio * (good_color_hsl[0] - bad_color_hsl[0])
    saturation = bad_color_hsl[1] + ratio * (good_color_hsl[1] - bad_color_hsl[1])
    lightness = bad_color_hsl[2] + ratio * (good_color_hsl[2] - bad_color_hsl[2])
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)

    badge_template = (
        Path(badge_template_path).read_text()
        if badge_template_path
        else DEFAULT_TEMPLATE
    )
    return eval(
        "f" + repr(badge_template),
        dict(
            coverage=f"{coverage_str}",
            color=f"#{''.join(f'{int(c * 255):02x}' for c in rgb)}",
            width=int(250 * len(coverage_str) / 3),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bad-color",
        type=str,
        default="hsl(9.62deg 71.56% 57.25%)",
    )
    parser.add_argument(
        "-g",
        "--good-color",
        type=str,
        default="hsl(103.64deg 84.62% 43.33%)",
    )
    parser.add_argument(
        "-m",
        "--min-coverage",
        type=int,
        default=70,
    )
    parser.add_argument(
        "-r",
        "--report-path",
        type=str,
        default="coverage.txt",
    )
    parser.add_argument(
        "-t",
        "--badge-template-path",
        type=str,
        default=None,
        nargs="?",
    )
    args = parser.parse_args()
    badge = make_badge(
        bad_color_hsl=parse_any_color_as_hsl(args.bad_color),
        good_color_hsl=parse_any_color_as_hsl(args.good_color),
        min_coverage=args.min_coverage,
        report_path=args.report_path,
        badge_template_path=args.badge_template_path,
    )
    print(badge)
