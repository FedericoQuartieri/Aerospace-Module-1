# Rende una serie di .pvti in una GIF.
#
#   pvpython scripts/paraview/make_gif.py <cartella> <uscita.gif> [campo] [slice|volume]
#
# esempio:
#   pvpython scripts/paraview/make_gif.py data/output/moving_sphere \
#            data/results/moving_sphere.gif velocity slice
from paraview.simple import *
import glob, os, subprocess, sys, tempfile

folder = sys.argv[1] if len(sys.argv) > 1 else "data/output/moving_sphere"
out    = sys.argv[2] if len(sys.argv) > 2 else "data/results/animazione.gif"
field  = sys.argv[3] if len(sys.argv) > 3 else "velocity"
mode   = sys.argv[4] if len(sys.argv) > 4 else "slice"
fps, width = 8, 900

files = sorted(glob.glob(os.path.join(folder, "sol_*.pvti")))
if not files:
    raise SystemExit(f"nessun sol_*.pvti in {folder}")

reader = XMLPartitionedImageDataReader(FileName=files)
reader.UpdatePipeline()
times = reader.TimestepValues

view = CreateRenderView()
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0      # via la terna XYZ in basso a sinistra
view.CameraParallelProjection = 1

if mode == "slice":
    b = reader.GetDataInformation().GetBounds()
    src = Slice(Input=reader)
    src.SliceType = 'Plane'
    src.SliceType.Origin = [(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2]
    src.SliceType.Normal = [0, 0, 1]
else:
    src = reader

# La finestra prende le proporzioni dei dati, altrimenti un dominio cubico
# viene tagliato sopra e sotto e un canale lascia margini vuoti ai lati.
bb = reader.GetDataInformation().GetBounds()
span_x, span_y = bb[1]-bb[0], bb[3]-bb[2]
aspect = (span_x / span_y) if span_y > 0 else 1.0
h = 780
view.ViewSize = [int(h * aspect) + 300, h]     # +300 per la barra dei colori

disp = Show(src, view)
disp.SetRepresentationType('Surface')
ColorBy(disp, ('POINTS', field, 'Magnitude') if field == 'velocity'
              else ('POINTS', field))
HideInteractiveWidgets(proxy=src)          # via il cerchio/piano interattivo

# la scala di colori si tara sull'ultimo istante: il primo puo' essere nullo
view.ViewTime = times[-1]
Render()

# Intervallo dei colori calcolato esplicitamente sull'ultimo istante.
# Le RescaleTransferFunctionToDataRange davano un intervallo degenere
# (0 .. 1e-38) sui campi che partono nulli, saturando tutto.
src.UpdatePipeline(times[-1])
info = src.GetPointDataInformation().GetArray(field)
lo, hi = info.GetComponentRange(-1 if info.GetNumberOfComponents() > 1 else 0)
if not (hi > lo):
    lo, hi = 0.0, 1.0
lut = GetColorTransferFunction(field)
lut.RescaleTransferFunction(lo, hi)
print(f"colour scale: {lo:.4g} .. {hi:.4g}")
lut.ApplyPreset('Viridis (matplotlib)', True)
disp.SetScalarBarVisibility(view, True)
sb = GetScalarBar(lut, view)
sb.TitleColor = [0, 0, 0]; sb.LabelColor = [0, 0, 0]
ResetCamera(); view.CameraParallelScale *= 0.62; Render()

tmp = tempfile.mkdtemp(prefix="pvgif_")
for i, t in enumerate(times):
    view.ViewTime = t
    Render()
    SaveScreenshot(os.path.join(tmp, f"f_{i:04d}.png"), view,
                   ImageResolution=view.ViewSize)
print(f"{len(times)} frames rendered")

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
pal = os.path.join(tmp, "pal.png")
vf = f"fps={fps},scale={width}:-1:flags=lanczos"
run = lambda c: subprocess.run(c, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
run(["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{tmp}/f_%04d.png",
     "-vf", vf + ",palettegen", pal])
run(["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{tmp}/f_%04d.png", "-i", pal,
     "-lavfi", vf + " [x]; [x][1:v] paletteuse", "-loop", "0", out])
print("written", out, f"({os.path.getsize(out)/1e6:.1f} MB)")
