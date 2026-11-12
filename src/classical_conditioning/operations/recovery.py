"""Publication-prefix decisions for the existing fixed-trace recovery stages."""
STAGES = ("intake", "corrected-preprocess", "candidate-metrics", "movement-state",
          "temporal-profiles", "trial-outcomes")

def dispositions(exists, *, stages=STAGES):
    """Require a contiguous prefix of wholly published stages."""
    result, missing = {}, False
    for name in stages:
        flags = exists[name]
        if any(flags) and not all(flags):
            raise ValueError("Partially published stage must be preserved for reconciliation: "+name)
        present = all(flags)
        if missing and present:
            raise ValueError("Published downstream stage has missing upstream stage: "+name)
        if not present: missing = True
        result[name] = "verify-existing" if present else "build-missing"
    return result

