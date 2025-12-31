# New Storage interface open issues

## Custom data 

Need more flexible way for doing this:
```python
s = StorageRegistry.get("qdb::quantlab")
r = s["QUBX", "METRICS"]

def metrics_ext(symbols, conditions, resample):
    conds = " and ".join(conditions)
    print(symbols)
    return f"""
    select timestamp, value, 'BTCUSDC' as symbol from qubx.metrics
    where {conds} and 
    strategy = 'kfs.statarb.live.delta.v01'
    and metric_name = 'total_capital' and is_live = true
    """

r = s["QUBX", "METRICS"]
r.add_external_builder("METRICS", metrics_ext)

d = r.read("BTCUSDC", "METRICS", "2025-11-01", "2025-12-01")
d.to_pd(True).plot()

```