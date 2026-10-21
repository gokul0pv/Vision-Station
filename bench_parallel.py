"""Does running a cycle's jobs in parallel help on this PC?

    python bench_parallel.py projects/chassis_job
    python bench_parallel.py chassis_app.vsp

Runs every job on its reference image, one after another and then side by side,
three times each, and reports the median cycle times -- and whether the results
are identical, which they must be. Turn on Run > Run Jobs in Parallel in
QuickBuild if it is clearly faster here. It cannot help on a single-core PC.
"""
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(path):
    import vsruntime
    app = vsruntime.load(path)
    jobs = [(jid, p, p.reference) for jid, (_n, p) in app._projects.items()]
    run = lambda j: (j[0], j[1].inspect(j[2], tools=j[1].build_tools()))
    for j in jobs:
        run(j)                                               # warm the engines up
    key = lambda res: {k: (r["verdict"], [(t["tool_id"], t["status"],
                                           json.dumps(t["values"], sort_keys=True, default=str))
                                          for t in r["tools"]]) for k, r in res.items()}
    n = min(len(jobs), os.cpu_count() or 1)
    seq_t, par_t, same = [], [], True
    for _ in range(3):
        t0 = time.perf_counter()
        seq = dict(run(j) for j in jobs)
        seq_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, n)) as ex:
            par = dict(ex.map(run, jobs))
        par_t.append(time.perf_counter() - t0)
        same = same and key(seq) == key(par)
    s, p = statistics.median(seq_t), statistics.median(par_t)
    print("%d jobs, %d cores" % (len(jobs), os.cpu_count() or 1))
    print("one after another: %.2f s" % s)
    print("side by side:      %.2f s   (%.0f%% of the time)" % (p, 100 * p / s))
    print("identical results: %s" % same)
    if not same:
        print("RESULTS DIFFER - leave parallel running off and report this.")
    elif p < 0.85 * s:
        print("Worth turning on: QuickBuild > Run > Run Jobs in Parallel.")
    else:
        print("No clear gain on this PC; leave it off.")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "projects",
                                                                       "chassis_job")))
