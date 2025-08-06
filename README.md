# ulx_analysis
Code used to analyze a catalog of ULX sources for UV data.

If the user has a raw catalog (no reddening corrections) then they should complete the following:
1. Run catalog_reduction.py on the user's original catalog so it is of a manageable size. 
2. Upload the resulting table to https://irsa.ipac.caltech.edu/applications/DUST/ and download the corresponding extinction table.
3. Run uv_analysis.py, which will combine the reduced catalog table and the extinction table into a usable form.

If the user has already included reddening data or the original catalog is small enough to be uploaded to the DUST website as-is, the user should simply run ulx.py.
