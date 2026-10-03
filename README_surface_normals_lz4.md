# Surface-normal LZ4 storage

`convert_to_lerobot2.sh --include-surface-normals` and
`convert_canonical_surface_normals_to_lz4.py` retain only the canonical,
losslessly verified LZ4 surface-normal chunks by default. The original MP4s
and metadata remain available for rollback while each split is committed.
After both the LZ4 manifest and metadata are committed successfully, the
temporary `_h264_backup` directory and `meta/info.json.h264_backup` are removed.
RGB/depth videos and depth PNG sidecars are unaffected.

Pass `--keep-h264-backup` to either command to explicitly retain the original
surface-normal MP4s and metadata. Split-parent conversions forward this option
to every split. Only retained backups are advertised in the LZ4 metadata and
manifest.

The standalone converter recognizes and validates previously installed LZ4
even when the optional H.264 backup has been deleted. Re-running it does not
recreate missing backups or clean up old backups on already-installed splits.
If cleanup fails after a successful commit, the installed LZ4 stays intact;
the error identifies the leftover paths for inspection. It never rolls back
to an H.264 directory that may have been partially removed.

Lightweight CPU fixtures (temporary directories only):

```bash
python -m unittest discover -s testing -p 'test_surface_normals_lz4_parallel.py'
bash -n convert_to_lerobot2.sh
```
