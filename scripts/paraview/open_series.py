# Apre una cartella di .pvti come UNA serie temporale.
#   paraview --script=scripts/paraview/open_series.py -- data/output/cavity
# Senza argomenti usa data/output/cavity.
from paraview.simple import *
import glob, os, sys

folder = "data/output/cavity"
for a in reversed(sys.argv[1:]):
    if not a.startswith("-") and os.path.isdir(a):
        folder = a
        break

files = sorted(glob.glob(os.path.join(folder, "sol_*.pvti")))
if not files:
    raise SystemExit(f"nessun sol_*.pvti in {folder}")

reader = XMLPartitionedImageDataReader(FileName=files,
                                       registrationName=os.path.basename(folder))
reader.UpdatePipeline()
times = reader.TimestepValues
print(f"{len(files)} files -> {len(times)} instants, from {times[0]} to {times[-1]}")

view = GetActiveViewOrCreate('RenderView')
disp = Show(reader, view)
disp.SetRepresentationType('Surface')
ColorBy(disp, ('POINTS', 'velocity', 'Magnitude'))
# scala di colori sull'ultimo istante, non sul primo che e' nullo
GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
view.ViewTime = times[-1]
Render()
disp.RescaleTransferFunctionToDataRange(False, True)
disp.SetScalarBarVisibility(view, True)
ResetCamera()
Render()
