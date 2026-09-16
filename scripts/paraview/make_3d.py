# Renders one instant of a .pvti series as a perspective view for the report.
#
#   pvpython scripts/paraview/make_3d.py <folder> <output.png> [options]
#
#   --frame N           which file of the series, -1 for the last (default)
#   --sections F [F ...]
#                       also cut the near half at these X, as fractions of
#                       the length (default none)
#   --seeds X Y0 Y1 Z0 Z1
#                       where streamlines start: the rectangle at X from
#                       (Y0, Z0) to (Y1, Z1), all fractions of the domain
#                       (default 0.02 0.1 0.9 0.6 0.9, near the inlet and in
#                       the near half)
#   --grid NY NZ        how many seeds along Y and Z (default 7 4, 0 0 for
#                       no streamlines)
#   --azimuth=DEGREES   turn the camera about Y, away from the +Z axis
#                       (default -30); pvpython mangles a negative number
#                       given as a separate argument, so keep the "="
#   --elevation=DEGREES then lift it (default 25)
#   --zoom FACTOR       above 1 moves the camera closer than the framing
#                       that fits the whole box (default 1), cropping it
#   --range LOW HIGH    fix the colour scale (default: the range of the cuts)
#   --width PIXELS      width of the image (default 2000)
#   --height PIXELS     height of the image (default 0.55 of the width)
#   --text PIXELS       size of the colour-bar labels (default 22); raise it
#                       for an image printed at half the page width
#
# example:
#   pvpython scripts/paraview/make_3d.py data/output/moving_sphere \
#            data/results/moving_sphere_3d.png --frame 10 --sections 0.5
#
# The domains of the report are symmetric about the middle Z plane, so the
# view shows only the near half: the middle plane and the sections coloured
# by the speed on the scale of make_still.py, the solid (where the
# permeability drops well below that of the fluid) cut by the middle plane
# and drawn in grey, grey too where a cut crosses it, and streamlines of the
# full 3D field started in the near half.  The cuts are unlit so their
# colours are the ones on the scale; the solid and the streamlines are lit,
# or their shape would not read.
from paraview.simple import *
import argparse
import glob
import math
import os

# The sequential blue of the charts, light to dark, as in make_still.py.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SOLID = "#c3c2b7"
LINES = "#e07b39"
MARGIN = 0.04


def rgb(hex_colour):
    return [int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]


def paint(display, colour, ambient, diffuse, specular=0.0):
    # Not ColorBy(display, None): it fails on a cut that misses the solid.
    display.ColorArrayName = ["POINTS", ""]
    display.AmbientColor = rgb(colour)
    display.DiffuseColor = rgb(colour)
    display.Ambient = ambient
    display.Diffuse = diffuse
    display.Specular = specular


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


parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("output")
parser.add_argument("--frame", type=int, default=-1)
parser.add_argument("--sections", type=float, nargs="+", default=[])
parser.add_argument("--seeds", type=float, nargs=5,
                    default=[0.02, 0.1, 0.9, 0.6, 0.9],
                    metavar=("X", "Y0", "Y1", "Z0", "Z1"))
parser.add_argument("--grid", type=int, nargs=2, default=[7, 4],
                    metavar=("NY", "NZ"))
parser.add_argument("--azimuth", type=float, default=-30.0)
parser.add_argument("--elevation", type=float, default=25.0)
parser.add_argument("--zoom", type=float, default=1.0)
parser.add_argument("--range", type=float, nargs=2, default=None,
                    metavar=("LOW", "HIGH"))
parser.add_argument("--width", type=int, default=2000)
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--text", type=int, default=22)
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

speed = Calculator(Input=reader)
speed.ResultArrayName = "speed"
speed.Function = "mag(velocity)"
fields = Calculator(Input=speed)
fields.ResultArrayName = "log_k"
fields.Function = "log10(permeability_X)"
fields.UpdatePipeline()
k_low, k_high = fields.PointData["log_k"].GetRange()
has_solid = k_high - k_low > 1.0
threshold = (k_low + k_high) / 2


def near_half(source):
    kept = Clip(Input=source)
    kept.ClipType = "Plane"
    kept.ClipType.Origin = centre
    kept.ClipType.Normal = [0.0, 0.0, 1.0]
    kept.Invert = 0
    kept.Crinkleclip = 0
    return kept


# The cuts: the middle plane, whole, and the sections, in the near half only.
cuts = []
for origin, normal in [(centre, [0.0, 0.0, 1.0])] + [
        ([low[0] + f * span[0], centre[1], centre[2]], [1.0, 0.0, 0.0])
        for f in args.sections]:
    cut = Slice(Input=fields)
    cut.SliceType = "Plane"
    cut.SliceType.Origin = origin
    cut.SliceType.Normal = normal
    cut.Triangulatetheslice = 0
    if normal[2] == 0.0:
        cut = near_half(cut)
    cut.UpdatePipeline()
    cuts.append((cut, origin, normal))

if args.range is not None:
    speed_low, speed_high = args.range
else:
    ranges = [cut.PointData["speed"].GetRange() for cut, _, _ in cuts]
    speed_low = min(r[0] for r in ranges)
    speed_high = max(r[1] for r in ranges)
if not speed_high > speed_low:
    speed_high = speed_low + 1.0

bar_strip = int(round(260 * args.text / 22))
height = args.height or int(round(args.width * 0.55))
view = CreateRenderView()
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0
view.UseFXAA = 1
# A narrow lens, so the far end of the channel does not shrink too much.
view.CameraViewAngle = 18
view.ViewSize = [args.width, height]

lut = GetColorTransferFunction("speed")
points = []
for index, colour in enumerate(RAMP):
    value = speed_low + (speed_high - speed_low) * index / (len(RAMP) - 1)
    points += [value] + rgb(colour)
lut.RGBPoints = points
lut.ColorSpace = "Lab"
# ParaView widens the scale to the data when the field is shown: keep the one
# chosen above.
lut.AutomaticRescaleRangeMode = "Never"

# Solid fills lie on their cut and are moved towards the camera once it is
# placed, so they are drawn on top instead of flickering in the same plane.
fills = []
for cut, origin, normal in cuts:
    shown = Show(cut, view)
    shown.SetRepresentationType("Surface")
    ColorBy(shown, ("POINTS", "speed"))
    shown.LookupTable = lut
    shown.Ambient = 1.0
    shown.Diffuse = 0.0
    shown.Specular = 0.0
    if has_solid:
        section = Clip(Input=cut)
        section.ClipType = "Scalar"
        section.Scalars = ["POINTS", "log_k"]
        section.Value = threshold
        section.Invert = 1
        section.Crinkleclip = 0
        moved = Transform(Input=section)
        fill = Show(moved, view)
        paint(fill, SOLID, 1.0, 0.0)
        fills.append((moved, origin, normal))
lut.RescaleTransferFunction(speed_low, speed_high)
GetDisplayProperties(cuts[0][0], view).SetScalarBarVisibility(view, True)

bar = GetScalarBar(lut, view)
bar.Title = "speed"
bar.ComponentTitle = ""
bar.HorizontalTitle = 1
bar.TitleColor = rgb(INK)
bar.LabelColor = rgb(INK_SECONDARY)
bar.TitleFontFamily = "Times"
bar.LabelFontFamily = "Times"
bar.TitleFontSize = int(round(args.text * 26 / 22))
bar.LabelFontSize = args.text
bar.UseCustomLabels = 1
bar.CustomLabels = nice_ticks(speed_low, speed_high)
bar.AddRangeLabels = 0
bar.LabelFormat = "%-#.2g"
bar.ScalarBarThickness = args.text
bar.ScalarBarLength = 0.6
bar.WindowLocation = "Any Location"
bar.Position = [(args.width - bar_strip + 40) / args.width, 0.2]

frame = Show(reader, view)
frame.SetRepresentationType("Outline")
paint(frame, INK_SECONDARY, 1.0, 0.0)

if has_solid:
    # The permeability jumps within one cell, so the raw surface is a
    # staircase: average the field over neighbouring cells twice, and smooth
    # the surface, before lighting it.
    blurred = fields
    for _ in range(2):
        cells = PointDatatoCellData(Input=blurred)
        cells.ProcessAllArrays = 0
        cells.PointDataArraytoprocess = ["log_k"]
        blurred = CellDatatoPointData(Input=cells)
        blurred.ProcessAllArrays = 0
        blurred.CellDataArraytoprocess = ["log_k"]
    body = Contour(Input=blurred)
    body.ContourBy = ["POINTS", "log_k"]
    body.Isosurfaces = [threshold]
    body.ComputeNormals = 0
    smooth = Smooth(Input=body)
    smooth.NumberofIterations = 300
    normals = GenerateSurfaceNormals(Input=smooth)
    solid = Show(near_half(normals), view)
    paint(solid, SOLID, 0.35, 0.65, 0.15)

ny, nz = args.grid
if ny > 0 and nz > 0:
    x, y0, y1, z0, z1 = [low[i] + f * span[i]
                         for i, f in zip((0, 1, 1, 2, 2), args.seeds)]
    seeds = Plane()
    seeds.Origin = [x, y0, z0]
    seeds.Point1 = [x, y1, z0]
    seeds.Point2 = [x, y0, z1]
    seeds.XResolution = max(ny - 1, 1)
    seeds.YResolution = max(nz - 1, 1)

    lines = StreamTracerWithCustomSource(Input=reader, SeedSource=seeds)
    lines.Vectors = ["POINTS", "velocity"]
    lines.IntegrationDirection = "BOTH"
    lines.MaximumStreamlineLength = 6.0 * max(span)
    # Near walls and inside solids the fluid is slow: without enough steps
    # a line stops there, and seems to end against the solid.
    lines.MaximumSteps = 50000
    tubes = Tube(Input=lines)
    tubes.Radius = 2.5e-3 * max(span)
    tubes.NumberofSides = 12
    traced = Show(tubes, view)
    paint(traced, LINES, 0.4, 0.6, 0.1)

# Start from the +Z side with Y up, then turn the camera.
Render(view)
view.CameraFocalPoint = centre
view.CameraPosition = [centre[0], centre[1], centre[2] + 4.0 * max(span)]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.ResetCamera(False)
camera = GetActiveCamera()
camera.Azimuth(args.azimuth)
camera.Elevation(args.elevation)
camera.OrthogonalizeViewUp()
view.ResetCamera(False)

# Fit the box in the image left of the colour bar.  In normalised device
# coordinates the image spans -1..1; dollying scales the projected box, the
# window centre shifts it.  Perspective ties the two, so repeat until both
# settle.
corners = [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
           for z in (low[2], high[2])]
aspect = args.width / height
right = 1.0 - 2.0 * bar_strip / args.width


def projected_box():
    matrix = camera.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
    xs, ys = [], []
    for corner in corners:
        point = matrix.MultiplyPoint(corner + [1.0])
        xs.append(point[0] / point[3])
        ys.append(point[1] / point[3])
    return min(xs), max(xs), min(ys), max(ys)


for _ in range(30):
    x_low, x_high, y_low, y_high = projected_box()
    ratio = max((x_high - x_low) / ((right + 1.0) * (1.0 - MARGIN)),
                (y_high - y_low) / (2.0 * (1.0 - MARGIN))) / args.zoom
    camera.Dolly(1.0 / ratio)
    x_low, x_high, y_low, y_high = projected_box()
    shift_x = (x_low + x_high) / 2 - (right - 1.0) / 2
    shift_y = (y_low + y_high) / 2
    centre_x, centre_y = camera.GetWindowCenter()
    camera.SetWindowCenter(centre_x + shift_x, centre_y + shift_y)

eye = camera.GetPosition()
for moved, origin, normal in fills:
    towards = sum((e - o) * n for e, o, n in zip(eye, origin, normal))
    lift = math.copysign(2e-3 * max(span), towards)
    moved.Transform.Translate = [lift * n for n in normal]
Render(view)

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
SaveScreenshot(args.output, view, ImageResolution=view.ViewSize)
print(f"{os.path.basename(path)}: speed {speed_low:.3g} .. {speed_high:.3g}, "
      f"solid {'yes' if has_solid else 'no'} -> {args.output}")
