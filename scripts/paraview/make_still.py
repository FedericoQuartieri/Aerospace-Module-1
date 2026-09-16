# Renders one instant of a .pvti series as a still for the report.
#
#   pvpython scripts/paraview/make_still.py <folder> <output.png> [options]
#
#   --normal x|y|z      the slice through the middle of the domain (default z)
#   --frame N           which file of the series, -1 for the last (default)
#   --seeds diagonal|inlet
#                       where streamlines start: across the slice, or along
#                       the X = 0 section for a channel (default diagonal)
#   --streamlines N     how many (default 24, 0 for none)
#   --width PIXELS      width of the domain in the image (default 1600)
#
# example:
#   pvpython scripts/paraview/make_still.py data/output/channel_obstacle \
#            data/results/channel_obstacle.png --seeds inlet
#
# The slice is coloured by the speed |u| on a single blue scale, light where
# the fluid is slow and dark where it is fast.  Where the permeability drops
# well below that of the fluid the cells are covered in grey with a dark
# outline: that is the solid.  Streamlines start in the slice and are traced
# in the full 3D field: the middle of these domains is a plane of symmetry, so
# they stay in it.  Nothing is lit, so every colour is the one on the scale.
from paraview.simple import *
import argparse
import glob
import math
import os

# The sequential blue of the charts, light to dark.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SOLID = "#c3c2b7"
BAR_STRIP = 260
MARGIN = 0.03


def rgb(hex_colour):
    return [int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]


def unlit(display, colour=None):
    display.Ambient = 1.0
    display.Diffuse = 0.0
    display.Specular = 0.0
    if colour is not None:
        ColorBy(display, None)
        display.AmbientColor = rgb(colour)
        display.DiffuseColor = rgb(colour)


def nice_ticks(low, high, count=4):
    raw = (high - low) / count
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next(f * magnitude for f in (1, 2, 2.5, 5, 10) if f * magnitude >= raw)
    first = math.ceil(low / step - 1e-9) * step
    ticks = []
    value = first
    while value <= high + 1e-9 * step:
        ticks.append(round(value, 12))
        value += step
    return ticks


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("output")
parser.add_argument("--normal", choices=["x", "y", "z"], default="z")
parser.add_argument("--frame", type=int, default=-1)
parser.add_argument("--seeds", choices=["diagonal", "inlet"],
                    default="diagonal")
parser.add_argument("--streamlines", type=int, default=24)
parser.add_argument("--width", type=int, default=1600)
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.folder, "sol_*.pvti")))
if not files:
    raise SystemExit(f"no sol_*.pvti in {args.folder}")
path = files[args.frame]

reader = XMLPartitionedImageDataReader(FileName=[path])
reader.UpdatePipeline()
bounds = reader.GetDataInformation().GetBounds()
low = [bounds[0], bounds[2], bounds[4]]
high = [bounds[1], bounds[3], bounds[5]]
centre = [(a + b) / 2 for a, b in zip(low, high)]
span = [b - a for a, b in zip(low, high)]

axis = "xyz".index(args.normal)
normal = [0.0, 0.0, 0.0]
normal[axis] = 1.0
# The two in-plane axes: `up` is drawn bottom to top, `across` sideways.
across, up = {0: (2, 1), 1: (0, 2), 2: (0, 1)}[axis]


def lifted(source, amount):
    """The same geometry moved towards the camera, so it is drawn on top of
    the slice instead of flickering in the same plane."""
    shift = [0.0, 0.0, 0.0]
    shift[axis] = amount * max(span)
    moved = Transform(Input=source)
    moved.Transform.Translate = shift
    return moved


cut = Slice(Input=reader)
cut.SliceType = "Plane"
cut.SliceType.Origin = centre
cut.SliceType.Normal = normal
cut.Triangulatetheslice = 0

speed = Calculator(Input=cut)
speed.ResultArrayName = "speed"
speed.Function = "mag(velocity)"
speed.UpdatePipeline()
speed_low, speed_high = speed.PointData["speed"].GetRange()
if not speed_high > speed_low:
    speed_high = speed_low + 1.0

log_k = Calculator(Input=cut)
log_k.ResultArrayName = "log_k"
log_k.Function = "log10(permeability_X)"
log_k.UpdatePipeline()
k_low, k_high = log_k.PointData["log_k"].GetRange()
has_solid = k_high - k_low > 1.0

# The domain fills the left part of the image, the colour bar the strip on
# the right.
domain_width = args.width
domain_height = min(int(round(args.width * span[up] / span[across])),
                    2 * args.width)
view = CreateRenderView()
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 1
view.UseFXAA = 1
view.ViewSize = [domain_width + BAR_STRIP, domain_height]

lut = GetColorTransferFunction("speed")
points = []
for index, colour in enumerate(RAMP):
    value = speed_low + (speed_high - speed_low) * index / (len(RAMP) - 1)
    points += [value] + rgb(colour)
lut.RGBPoints = points
lut.ColorSpace = "Lab"

shown = Show(speed, view)
shown.SetRepresentationType("Surface")
ColorBy(shown, ("POINTS", "speed"))
shown.LookupTable = lut
unlit(shown)
shown.SetScalarBarVisibility(view, True)

bar = GetScalarBar(lut, view)
bar.Title = "speed"
bar.ComponentTitle = ""
bar.HorizontalTitle = 1
bar.TitleColor = rgb(INK)
bar.LabelColor = rgb(INK_SECONDARY)
bar.TitleFontFamily = "Times"
bar.LabelFontFamily = "Times"
bar.TitleFontSize = 26
bar.LabelFontSize = 22
bar.UseCustomLabels = 1
bar.CustomLabels = nice_ticks(speed_low, speed_high)
bar.AddRangeLabels = 0
bar.LabelFormat = "%-#.2g"
bar.ScalarBarThickness = 22
bar.ScalarBarLength = 0.7
bar.WindowLocation = "Any Location"
bar.Position = [(domain_width + 40) / (domain_width + BAR_STRIP), 0.12]

frame = Show(cut, view)
frame.SetRepresentationType("Outline")
unlit(frame, SOLID)

if has_solid:
    # Cut at the same level as the outline, so the grey ends exactly on it
    # instead of on the staircase of whole cells a threshold would keep.
    threshold = (k_low + k_high) / 2
    solid = Clip(Input=log_k)
    solid.ClipType = "Scalar"
    solid.Scalars = ["POINTS", "log_k"]
    solid.Value = threshold
    solid.Invert = 1
    solid.Crinkleclip = 0
    # Above the streamlines: the small velocity the drag leaves inside the
    # solid is not flow worth drawing.
    fill = Show(lifted(solid, 4e-3), view)
    fill.SetRepresentationType("Surface")
    unlit(fill, SOLID)

    edge = Contour(Input=log_k)
    edge.ContourBy = ["POINTS", "log_k"]
    edge.Isosurfaces = [threshold]
    outline = Show(lifted(edge, 5e-3), view)
    unlit(outline, INK)
    outline.LineWidth = 2.0

if args.streamlines > 0:
    lines = StreamTracer(Input=reader, SeedType="Line")
    lines.Vectors = ["POINTS", "velocity"]
    lines.IntegrationDirection = "BOTH"
    lines.MaximumStreamlineLength = 4.0 * max(span)
    start = list(centre)
    end = list(centre)
    if args.seeds == "inlet":
        start[across] = end[across] = low[across] + 0.02 * span[across]
        start[up] = low[up] + 0.02 * span[up]
        end[up] = high[up] - 0.02 * span[up]
    else:
        start[across], start[up] = (low[across] + 0.02 * span[across],
                                    low[up] + 0.02 * span[up])
        end[across], end[up] = (high[across] - 0.02 * span[across],
                                high[up] - 0.02 * span[up])
    lines.SeedType.Point1 = start
    lines.SeedType.Point2 = end
    lines.SeedType.Resolution = args.streamlines
    traced = Show(lifted(lines, 2e-3), view)
    unlit(traced, INK_SECONDARY)
    # Thin, nearly horizontal lines rasterise as steps whose joins vanish on
    # the dark end of the scale and read as dashes: two pixels, antialiased.
    traced.LineWidth = 2.0

# paraview.simple resets the camera on the first render of a view, which would
# undo the framing below: let that happen first.
Render(view)

# Camera on the +normal side, `up` pointing up.  Half the view height is the
# half-height of the domain plus a margin; the focal point moves right by half
# the colour-bar strip so the domain sits on the left.
view_up = [0.0, 0.0, 0.0]
view_up[up] = 1.0
right = cross(view_up, normal)
scale = 0.5 * span[up] * (1.0 + MARGIN)
world_per_pixel = 2.0 * scale / domain_height
# The domain must also fit across; widen the scale if it does not.
if span[across] / world_per_pixel > domain_width / (1.0 + MARGIN):
    world_per_pixel = span[across] * (1.0 + MARGIN) / domain_width
    scale = 0.5 * world_per_pixel * domain_height
offset = 0.5 * BAR_STRIP * world_per_pixel
focal = [c + r * offset for c, r in zip(centre, right)]
position = list(focal)
position[axis] += 10.0 * max(span)
view.CameraFocalPoint = focal
view.CameraPosition = position
view.CameraViewUp = view_up
view.CameraParallelScale = scale
Render(view)

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
SaveScreenshot(args.output, view, ImageResolution=view.ViewSize)
print(f"{os.path.basename(path)}: speed {speed_low:.3g} .. {speed_high:.3g}, "
      f"solid {'yes' if has_solid else 'no'} -> {args.output}")
