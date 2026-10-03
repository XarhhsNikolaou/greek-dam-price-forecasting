Test year July 2025 - June 2026, general model retrained quarterly. MAE with all features: **18.09 EUR/MWh**.

| Group | Built-in importance (gain) | MAE rise when scrambled | MAE without the group (retrained) | Error reduction from adding the group |
|---|---|---|---|---|
| Fundamentals | 43% | +17.32 | 20.55 (+2.46) | 12% |
| Price history | 54% | +12.94 | 22.01 (+3.92) | 18% |
| Calendar | 3% | +0.60 | 18.55 (+0.46) | 2% |

Single fundamental features, MAE rise when scrambled:

- `net_load_total_mw`: +11.40
- `net_load_proxy_mw`: +1.68
- `hydro_reservoir_mwh`: +0.37
- `res_forecast_mw`: +0.32
- `NGAS_Price`: +0.26
- `hydro_reservoir_change`: +0.11
- `load_forecast_mw`: +0.11
- `net_out_forecast`: +0.06
- `carbon_price_eur`: -0.08
