$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$here\prediction_tracker_feeder.py" --bucket sharp-models
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
