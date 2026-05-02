#!/usr/bin/env python3
"""Expands seed data to full dataset. Run: python generate_dataset.py --seed-dir dataset --out expanded"""
import json, os, copy, argparse, shutil
from pathlib import Path

def load(fp): return json.load(open(fp))
def save(fp, data):
    os.makedirs(fp.parent, exist_ok=True)
    json.dump(data, open(fp, "w"), indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", default=str(Path(__file__).parent), help="Directory containing seed JSONs")
    parser.add_argument("--out", default=str(Path(__file__).parent), help="Output directory for expanded dataset")
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    out_dir = Path(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Copy categories directory
    cat_src = seed_dir / "categories"
    cat_dst = out_dir / "categories"
    if cat_src.exists() and cat_src != cat_dst:
        if cat_dst.exists():
            shutil.rmtree(cat_dst)
        shutil.copytree(cat_src, cat_dst)

    merchants_seed = load(seed_dir / "merchants_seed.json")["merchants"]
    customers_seed = load(seed_dir / "customers_seed.json")["customers"]
    triggers_seed  = load(seed_dir / "triggers_seed.json")["triggers"]

    # Expand merchants
    out_merchants = {}
    for m in merchants_seed:
        out_merchants[m["merchant_id"]] = m
    for m in merchants_seed:
        for suffix, vm, cm in [("b",0.6,0.5),("c",1.4,1.3),("d",0.3,0.2),("e",2.0,1.8)]:
            nm = copy.deepcopy(m)
            nm["merchant_id"] = m["merchant_id"] + "_" + suffix
            p = nm["performance"]
            p["views"] = int(p["views"] * vm)
            p["calls"] = int(p["calls"] * cm)
            out_merchants[nm["merchant_id"]] = nm
    
    # Save individual and merged
    mdir = out_dir / "merchants"
    for mid, m in out_merchants.items():
        save(mdir / f"{mid}.json", m)
    save(out_dir / "merchants_seed.json", {"merchants": list(out_merchants.values())})
    print(f"[OK] {len(out_merchants)} merchants")

    # Expand customers
    out_customers = {}
    for c in customers_seed:
        out_customers[c["customer_id"]] = c
    for c in customers_seed:
        for j, state in enumerate(["active","lapsed_soft","lapsed_hard","new"]):
            nc = copy.deepcopy(c)
            nc["customer_id"] = f"{c['customer_id']}_v{j}"
            nc["state"] = state
            out_customers[nc["customer_id"]] = nc
            
    cdir = out_dir / "customers"
    for cid, c in out_customers.items():
        save(cdir / f"{cid}.json", c)
    save(out_dir / "customers_seed.json", {"customers": list(out_customers.values())})
    print(f"[OK] {len(out_customers)} customers")

    # Expand triggers
    out_triggers = {t["id"]: t for t in triggers_seed}
    kinds = ["dormant_with_vera","curious_ask_due","gbp_unverified","perf_spike","milestone_reached"]
    for i, mid in enumerate(list(out_merchants.keys())[:15]):
        kind = kinds[i % len(kinds)]
        tid = f"trg_gen_{i:03d}_{kind}"
        out_triggers[tid] = {"id":tid,"scope":"merchant","kind":kind,"source":"internal",
                             "merchant_id":mid,"customer_id":None,"payload":{"merchant_id":mid},
                             "urgency":(i%5)+1,"suppression_key":f"{kind}:{mid}:2026",
                             "expires_at":"2026-06-30T00:00:00Z"}
                             
    tdir = out_dir / "triggers"
    for tid, t in out_triggers.items():
        save(tdir / f"{tid}.json", t)
    save(out_dir / "triggers_seed.json", {"triggers": list(out_triggers.values())})
    print(f"[OK] {len(out_triggers)} triggers\nDone!")

if __name__ == "__main__":
    main()
