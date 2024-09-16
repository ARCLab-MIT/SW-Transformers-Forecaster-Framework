# Data Contents 📊

This folder contains all the data files used during the implementation. Below, we highlight the main files (some files represent the same data with different formats and are not used).

## Indices Data ☀️🌍
- [`SOLFSMY.TXT`](/data/SOLFSMY.TXT): Contains F10.7, S10.7, M10.7, and Y10.7 indices data back to 1997, extracted from the [SET webpage](https://spacewx.com/jb2008/).
- [`SOLF107_Historical.TXT`](/data/SOLF107_HistoricalValues.txt): F10.7 data dating back to 1947, extracted from [LISIRD NOAA](https://lasp.colorado.edu/lisird/data/noaa_radio_flux).
- [`SET_SOLFSMY.TXT`](/data/SET_SOLFSMY.TXT): Corrected indices from SET used to reclassify our data to match their benchmark ([Licata et al., 2020](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2020SW002496)).
- [`DST_IAGA2002.txt`](/data/DST_IAGA2002.txt): Dst geomagnetic index data back to 1957, extracted from the [WDC of Kyoto](https://wdc.kugi.kyoto-u.ac.jp/dstae/index.html). The other DST file has the same data in a more difficult-to-process format from WDC.
- [`SOLAP.TXT`](/data/SOLAP.TXT): Ap geomagnetic index data back to 1932, extracted from the [GFZ Potsdam](https://www.gfz-potsdam.de/en/section/geomagnetism/data-products-services/geomagnetic-kp-index).

## Benchmark Data 📈

All this data is extracted from the paper by [Licata et al. (2020)](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2020SW002496).
- [`paper_results.csv`](/data/paper_results.csv): MSE and STPE obtained by the authors for FSMY 10.7 indices forecasting.
- [`benchmark_results_ap.csv`](/data/benchmark_results_ap.csv): MSE and STPE results for Ap index forecasting.
- [`benchmark_results_dst.csv`](/data/benchmark_results_dst.csv): MSE and STPE results for Dst index forecasting.
