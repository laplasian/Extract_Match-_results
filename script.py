import re, os, sys, yaml

def read_all(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_hkl_precise(hkl_text):
    phases = {}
    header_iter = list(re.finditer(r"Pattern#\s*\d+\s+Phase\s+No\.\s*:\s*\d+\s+([A-Za-z0-9_\-\(\)\[\]]+).*", hkl_text))
    for idx, m in enumerate(header_iter):
        name = m.group(1).strip()
        start = m.end()
        end = header_iter[idx+1].start() if idx+1 < len(header_iter) else len(hkl_text)
        block = hkl_text[start:end]
        cell_m = re.search(r"CELL:\s*([0-9\.\s]+)", block)
        lattice = None
        if cell_m:
            parts = [float(x) for x in cell_m.group(1).split()[:6]]
            if len(parts) == 6:
                lattice = {"a": parts[0], "b": parts[1], "c": parts[2], "alpha": parts[3], "beta": parts[4], "gamma": parts[5]}
        refls = []
        for line in block.splitlines():
            s = line.strip()
            if not s or s.startswith(("!", "Code")):
                continue
            cols = line.split()
            if len(cols) >= 11 and cols[0].isdigit():
                try:
                    h, k, l = int(cols[1]), int(cols[2]), int(cols[3])
                    d = float(cols[5]); two_theta = float(cols[6]); Iobs = float(cols[8]); Icalc = float(cols[9])
                    refls.append({"hkl":[h,k,l], "two_theta":two_theta, "d_spacing":d, "I_obs":Iobs, "I_calc":Icalc})
                except Exception:
                    pass
        phases[name] = {}
        if lattice: phases[name]["lattice"] = lattice
        if refls: phases[name]["reflections"] = refls
    return phases

def parse_fullprof(base):
    sum_text = read_all(base + ".sum")
    out_text = read_all(base + ".out")
    hkl_text = read_all(base + ".hkl")
    data = {"phases": {}}
    # Phase names
    phases = re.findall(r"Phase\s+No\.\s*:\s*\d+\s+(?:Phase\s+name:\s*)?([A-Za-z0-9_\-\(\)\[\] ]+)", sum_text + out_text)
    phases = [p.strip() for p in phases if p.strip()]
    if not phases and hkl_text:
        phases = list(parse_hkl_precise(hkl_text).keys())
    if not phases:
        phases = ["Phase1"]
    seen=set(); uniq_phases=[]
    for p in phases:
        pn = p.split()[0]
        if pn not in seen:
            seen.add(pn); uniq_phases.append(pn)
    for ph in uniq_phases:
        data["phases"][ph] = {}
    # Weight fractions
    m = re.findall(r"=>\s*Phase:\s*\d+\s+([^\n\r]+).*?\n.*?Fract\(%\):\s*([\d\.]+)(?:\(([\d\.]+)\))?", sum_text, re.DOTALL)
    if not m:
        m = re.findall(r"=>\s*Phase:\s*\d+\s+([^\n\r]+).*?\n.*?Fract\(%\):\s*([\d\.]+)(?:\(([\d\.]+)\))?", out_text, re.DOTALL)
    if m:
        for name, val, err in m:
            name = name.strip().split()[0]
            data["phases"].setdefault(name, {})["weight_fraction"] = f"{float(val):.2f}" + (f" ± {err}" if err else "")
    elif len(uniq_phases)==1:
        bare = re.findall(r"Fract\(%\):\s*([\d\.]+)(?:\(([\d\.]+)\))?", sum_text + out_text)
        if bare:
            val, err = bare[-1]
            data["phases"][uniq_phases[0]]["weight_fraction"] = f"{float(val):.2f}" + (f" ± {err}" if err else "")
    # Lattice + reflections from hkl
    if hkl_text:
        hkl_ph = parse_hkl_precise(hkl_text)
        for name, pd in hkl_ph.items():
            data["phases"].setdefault(name, {}).update(pd)
    # Microstrain
    strain_vals = []
    for txt in (sum_text, out_text):
        for m in re.finditer(r"Strain parameters\s*:([^\n\r]+)\n([^\n\r]+)?\n?([^\n\r]+)?", txt, re.IGNORECASE):
            nums = []
            for g in m.groups():
                if not g: continue
                for tok in g.split():
                    try: nums.append(float(tok))
                    except Exception: pass
            nz = [v for v in nums if abs(v)>1e-12]
            if nz: strain_vals = nz; break
        if strain_vals: break
    if strain_vals:
        for ph in data["phases"]:
            data["phases"][ph]["microstrain"] = strain_vals if len(strain_vals)>1 else strain_vals[0]
    else:
        for ph in data["phases"]:
            data["phases"][ph]["microstrain"] = {"value": 0.0, "note": "not refined or reported as zeros"}
    # Metrics
    rp_blocks = re.findall(r"Rp:\s*([\d\.]+)\s+Rwp:\s*([\d\.]+)\s+Rexp:\s*([\d\.]+)\s+Chi2:\s*([\d\.]+)", sum_text)
    metrics = {}
    if rp_blocks:
        Rp,Rwp,Rexp,Chi2 = rp_blocks[-1]
        metrics.update({"Rp": float(Rp), "Rwp": float(Rwp), "Rexp": float(Rexp), "Chi2": float(Chi2)})
    else:
        rp_blocks = re.findall(r"R-Factors:\s*([\d\.]+)\s+([\d\.]+)\s+Chi2:\s*([\d\.]+)", out_text)
        if rp_blocks:
            Rp,Rwp,Chi2 = rp_blocks[-1]
            metrics.update({"Rp": float(Rp), "Rwp": float(Rwp), "Chi2": float(Chi2)})
    # Per-phase Bragg/RF from out
    per_phase = re.findall(r"Pattern#\s*\d+.*?Phase\s+No\.\s*:\s*\d+\s+([^\n\r]+).*?Bragg R-factor:\s*([\d\.]+).*?RF-factor\s*[:=]\s*([\d\.]+)", out_text, re.DOTALL)
    for name, br, rf in per_phase:
        name = name.strip().split()[0]
        data["phases"].setdefault(name, {})
        data["phases"][name]["Bragg_R_factor"] = float(br)
        data["phases"][name]["RF_factor"] = float(rf)
    if metrics:
        data["metrics"] = metrics
    return data

def extract_fullprof_results(base):
    res = parse_fullprof(base)
    out_yaml = f"{base}_results.yml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(res, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote: {out_yaml}")

extract_fullprof_results(R"C:\Users\user\Downloads\WORK\tmp\fpcalc")
