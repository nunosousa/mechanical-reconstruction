# Measurements

Record the coordinate-system definition and use it consistently.

For every measured landmark, record whether it is a hole centre, tube centre, surface edge, inferred centre, etc. Also record measurement precision or uncertainty where useful.

Do not silently mix coordinate systems or units.

The 3D landmarks CSV uses `name,x,y,z` columns and may include a `circular`
column. Set it to `true` for landmarks whose image position should be fitted
from a circular edge; use `false` or leave it blank for ordinary point clicks.
