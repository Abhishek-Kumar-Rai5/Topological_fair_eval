"""E4 (CTU-13) on toy flows: windows, labels, graphs, zigzag, folds. No real data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import e4, ctu13
from src import features_ctu13 as fc
from src.metadata import is_valid_result

INF, NORM = "10.0.0.1", "10.0.0.2"


def write_scenario(root, scen, seed=0, n_bins=400, with_botnet=True):
    """Toy capture: one infected host (botnet flows in bins 100-260), one normal host, one background host."""
    rng = np.random.default_rng(seed)
    folder = Path(root) / str(ctu13.FIRST_CAPTURE + scen - 1)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "README").write_text(f"Infected Machines:\nWindows XP Name: X, IP: {INF} (Label: Botnet)\n")
    rows = []
    t0 = pd.Timestamp("2011/08/10 10:00:00")
    for b in range(n_bins):
        for host, label in ((NORM, "flow=From-Normal-V42-x"), ("10.0.0.9", "flow=Background")):
            for _ in range(rng.integers(1, 4)):
                rows.append((t0 + pd.Timedelta(seconds=10 * b + rng.random() * 9), host,
                             f"192.168.0.{rng.integers(1, 6)}", str(rng.choice([80, 443])), label))
        bot = with_botnet and 100 <= b < 260
        for _ in range(rng.integers(1, 3) if not bot else rng.integers(5, 9)):
            rows.append((t0 + pd.Timedelta(seconds=10 * b + rng.random() * 9), INF,
                         f"172.16.{rng.integers(0, 40)}.{rng.integers(0, 40)}" if bot else "192.168.0.1",
                         str(rng.integers(1, 3000)) if bot else "80",
                         "flow=From-Botnet-V42-x" if bot else "flow=Background"))
    rows.sort()
    df = pd.DataFrame({"StartTime": [r[0].strftime("%Y/%m/%d %H:%M:%S.%f") for r in rows], "Dur": 0.1, "Proto": "tcp",
                       "SrcAddr": [r[1] for r in rows], "Sport": 1000, "Dir": "->", "DstAddr": [r[2] for r in rows],
                       "Dport": [r[3] for r in rows], "State": "S", "sTos": 0, "dTos": 0, "TotPkts": 3, "TotBytes": 300,
                       "SrcBytes": 100, "Label": [r[4] for r in rows]})
    df.to_csv(folder / "capture.binetflow", index=False)


def cfg(root, seeds=(0, 1)):
    c = yaml.safe_load(Path("configs/e4_ctu13.yaml").read_text())
    c.update(data_root=str(root), seeds=list(seeds))
    c["detector"]["n_estimators"] = 30
    return c


# ---------- reading, windows, labels ----------
def test_read_scenario_hosts_bins_and_labels(tmp_path):
    write_scenario(tmp_path, 1)
    sc = ctu13.read_scenario(ctu13.scenario_folder(1, tmp_path))
    assert sc.infected == [INF] and sc.normal == [NORM]                   # background host is not a listed host
    assert set(sc.flows["host"]) == {INF, NORM}
    assert sc.flows["bin"].min() >= 0 and sc.n_bins >= 399
    assert set(sc.flows["cls"]) == {"From-Botnet", "Background", "From-Normal"}


def test_window_grid_and_label_rules(tmp_path):
    write_scenario(tmp_path, 1)
    sc = ctu13.read_scenario(ctu13.scenario_folder(1, tmp_path))
    hf = sc.flows[sc.flows.host == INF]
    js = ctu13.window_range(hf, sc.n_bins)
    assert js[0] == 0 and (30 * js + 59 <= sc.n_bins - 1).all()           # step 30, full windows only
    labels = {int(j): ctu13.window_label(ctu13.window_flows(hf, j), True) for j in js}
    assert labels[0] == -1                                                # bins 0-59: background only -> excluded
    assert labels[5] == 1                                                 # bins 150-209: botnet flows
    nf = sc.flows[sc.flows.host == NORM]
    assert ctu13.window_label(ctu13.window_flows(nf, 2), False) == 0
    bad = ctu13.window_flows(hf, 5)                                       # a normal host that sends botnet flows
    assert ctu13.window_label(bad, False) == -1


def test_channel_matrix_values():
    wf = pd.DataFrame({"host": "h", "bin": [30, 30, 31], "dst": ["a", "b", "a"], "dport": ["80", "80", "81"],
                       "pkts": [1.0, 2, 3], "bytes": [10.0, 20, 30], "src_bytes": [5.0, 5, 5], "cls": "x"})
    m = ctu13.channel_matrix(wf, 1)                                       # window 1 starts at bin 30
    assert m.shape == (60, 6)
    assert m[0].tolist() == [2, 3, 30, 10, 2, 1] and m[1].tolist() == [1, 3, 30, 5, 1, 1] and m[2:].sum() == 0


def test_window_graph_cap_keeps_most_flows():
    wf = pd.DataFrame({"dst": [f"i{k}" for k in range(10)] + ["i0"] * 5, "dport": ["80"] * 15})
    edges, capped = ctu13.window_graph(wf, max_edges=3)
    assert capped and len(edges) == 3 and edges[0][0] == "i0"             # i0:80 has 6 flows
    edges, capped = ctu13.window_graph(wf)
    assert not capped and len(edges) == 10


# ---------- graph features ----------
def test_graph_stats():
    c4 = [("a", "port:1"), ("a", "port:2"), ("b", "port:1"), ("b", "port:2")]      # 4-cycle
    s = fc.graph_stats(c4)
    assert (s["V"], s["E"], s["components"], s["cycle_rank"], s["diameter"]) == (4, 4, 1, 1, 2)
    two = fc.graph_stats([("a", "port:1"), ("b", "port:2")])
    assert (two["components"], two["cycle_rank"], two["diameter"]) == (2, 0, 1)
    assert fc.graph_stats([])["V"] == 0
    assert fc.z_control(c4).tolist() == [4, 4, 1, 2] and len(fc.f2_graph_features(c4)) == 5
    path = [(f"i{k}", f"port:{k}") for k in range(400)] + [(f"i{k + 1}", f"port:{k}") for k in range(400)]
    assert fc.graph_stats(path)["diameter"] >= 700                         # large path: double sweep ~ 800


# ---------- zigzag ----------
TRI = [("a", "b"), ("b", "c"), ("a", "c")]
PATH = [("a", "b"), ("b", "c")]


def h1(d):
    return sorted(map(tuple, d.tolist()))


def test_zigzag_hand_example_cycle_in_windows_0_and_1():
    d0, d1, n = fc.host_zigzag([TRI, TRI, PATH])
    assert n == 3 and h1(d1) == [(0.0, 4.0)]                              # the loop lives in X0..X3
    assert h1(d0) == [(0.0, float("inf"))]


def test_zigzag_simultaneous_removal_and_gap_do_not_crash():
    d0, d1, n = fc.host_zigzag([TRI, [], TRI, PATH, []])
    assert len(d1) == 2                                                   # two separate loops (windows 0 and 2)


def test_z_features_use_only_the_past():
    graphs = [TRI, TRI, PATH, TRI, PATH, PATH, TRI, []]
    full = fc.zigzag_features(*fc.host_zigzag(graphs))
    for t in range(1, len(graphs)):
        trunc = fc.zigzag_features(*fc.host_zigzag(graphs[:t + 1]))
        assert np.allclose(full[t], trunc[t]), t                          # window t does not see windows > t


def test_z_feature_shape_and_names():
    z = fc.zigzag_features(*fc.host_zigzag([TRI, TRI, PATH]))
    assert z.shape == (3, 14) and len(fc.z_names()) == 14 and np.isfinite(z).all()


# ---------- stage 1 and 2 ----------
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("ctu")
    out = tmp_path_factory.mktemp("out")
    for s in (1, 2, 3, 4):
        write_scenario(root, s, seed=s, with_botnet=(s != 4))
    c = cfg(root)
    for s in (1, 2, 3, 4):
        e4.build_scenario(c, s, out)
    return c, out


def test_stage1_table_columns_labels_and_validity(built):
    c, out = built
    p = out / "by_scenario" / "1.parquet"
    assert is_valid_result(p)
    t = pd.read_parquet(p)
    cols = e4.feature_columns()
    assert len(cols["F1"]) == 29 and len(cols["Z"]) == 14 and len(cols["ZC"]) == 4 and len(cols["G"]) == 5
    assert set(t["label"]) == {0, 1} and set(t["host_type"]) == {"infected", "normal"}
    assert (t.loc[t.host_type == "infected", "label"] == 1).all()
    assert t[sum(cols.values(), [])].notna().all().all()
    assert t.loc[t.label == 1, "zc_edges"].mean() > t.loc[t.label == 0, "zc_edges"].mean()   # toy scan has more edges


def test_scenario_without_positives_is_not_a_fold(built):
    c, out = built
    t4 = pd.read_parquet(out / "by_scenario" / "4.parquet")
    assert not (t4["label"] == 1).any()
    assert e4.usable_scenarios(c) == [1, 2, 3, 4]


def test_validation_draw_by_seed_excludes_test():
    for seed in range(5):
        v = e4.draw_validation(seed, 3, list(range(1, 13)))
        assert v != 3 and v == e4.draw_validation(seed, 3, list(range(1, 13)))
    assert len({e4.draw_validation(s, 3, list(range(1, 13))) for s in range(20)}) > 1


def test_fold_rows_and_rules(built):
    c, out = built
    tables = e4.load_tables(out, [1, 2, 3, 4])
    df = e4.run_fold(c, tables, 1)
    assert set(df.feature_set) == set(e4.SETS) and len(df) == 4 * 2
    assert (df.scenario == 1).all() and (df.validation_scenario != 1).all()
    assert (df.n_positive > 0).all() and df.pr_auc.between(0, 1).all()
    assert (df.val_alarm_rate <= 0.05).all()                              # threshold from validation, not test
    assert df.false_alarms_per_day.notna().all()
    assert (df.groupby("seed").n_train_windows.first() > 0).all()


def test_train_never_contains_test_or_validation_scenario(built, monkeypatch):
    c, out = built
    tables = e4.load_tables(out, [1, 2, 3, 4])
    seen = []
    orig = e4.F3Features.fit
    monkeypatch.setattr(e4.F3Features, "fit", lambda self, w: (seen.append(len(w)), orig(self, w))[1])
    df = e4.run_fold(c, tables, 2)
    for seed, n in zip(c["seeds"], seen):
        v = e4.draw_validation(seed, 2, [1, 2, 3, 4])
        expect = sum((tables[s].label == 0).sum() for s in (1, 2, 3, 4) if s not in (2, v))
        assert n == expect                                                # normal windows of the other scenarios only


def test_fold_save_merge_and_resume(built, tmp_path):
    c, out = built
    tables = e4.load_tables(out, [1, 2, 3, 4])
    e4.run_fold_and_save(c, tables, 1, tmp_path)
    assert is_valid_result(tmp_path / "by_fold" / "1.csv")
    path, _ = e4.merge_results(c, [1], tmp_path)
    assert is_valid_result(path)
    (tmp_path / "by_fold" / "1.csv").write_text("")
    assert not is_valid_result(tmp_path / "by_fold" / "1.csv")


def test_summary_script_on_toy_results():
    sys.path.insert(0, "scripts")
    pytest.importorskip("summarize_smd", reason="scripts/summarize_smd.py comes from master")
    import summarize_e4 as s4
    rng = np.random.default_rng(0)
    rows = [{"machine": f"scenario-{s}", "scenario": s, "seed": sd, "feature_set": fs, "pr_auc": 0.5 + 0.1 * (fs == "Z") + rng.normal(0, .01),
             "roc_auc": .8, "recall_at_thr": .5, "false_alarms_per_day": 1., "runtime_per_window_ms": 1., "val_alarm_rate": .01,
             "n_positive": 40, "n_normal": 100} for s in range(1, 9) for sd in (0, 1) for fs in e4.SETS]
    summary, v, pos = s4.verdict_table(pd.DataFrame(rows), 0)
    assert len(summary) == 4 and len(pos) == 8
    z = v[v.comparison == "Z - Z-control"].iloc[0]
    assert z.verdict == "Helps" and z.delta_pr_auc > 0.05


def test_f3_split_sets_are_the_halves_of_the_same_f3(built):
    """E4x (exploratory): H0-only and H1-only add 16 columns each to F1+F2; the full F3 adds 32; E4 itself is unchanged."""
    c, out = built
    tables = e4.load_tables(out, [1, 2, 3, 4])
    cfg = dict(c, feature_sets=["F1+F2", "F1+F2+F3", "F1+F2+F3-H0", "F1+F2+F3-H1"])
    df = e4.run_fold(cfg, tables, 1)
    n = df.groupby("feature_set").n_features.first()
    assert n["F1+F2+F3-H0"] == n["F1+F2"] + 16 and n["F1+F2+F3-H1"] == n["F1+F2"] + 16 and n["F1+F2+F3"] == n["F1+F2"] + 32
    base = e4.run_fold(c, tables, 1)                                       # the E4 sets alone
    a = df[df.feature_set == "F1+F2+F3"].sort_values("seed").pr_auc.to_numpy()
    b = base[base.feature_set == "F1+F2+F3"].sort_values("seed").pr_auc.to_numpy()
    assert np.allclose(a, b)                                               # adding the split changes nothing in E4


def test_e4x_summary_script_on_toy_results():
    sys.path.insert(0, "scripts")
    pytest.importorskip("summarize_smd")
    import summarize_e4x as sx
    assert ("F1+F2+F3-H1", "F1+F2+F3-H0") in sx.COMPARISONS


def test_flow_label_strings_name_behaviours():
    code = ctu13.behaviour_code
    assert code("flow=From-Botnet-V42-TCP-CC6-Plain-HTTP-Encrypted-Data") == ctu13.SHAPE_BIT
    assert code("flow=From-Botnet-V42-UDP-CC108-Established-P2P-Not-Encrypted-1") == ctu13.SHAPE_BIT
    assert code("flow=From-Botnet-V42-TCP-Attempt-SPAM") == ctu13.IDENTITY_BIT
    assert code("flow=From-Botnet-V42-UDP-DNS") == 0 and code("flow=Background-TCP-Attempt") == 0
    assert code("flow=From-Botnet-V50-1-UDP-DNS") == 0                    # a '1' alone is not C&C
    assert code("flow=From-Botnet-V42-TCP-Established-HTTP-Ad-5") == 0    # ad traffic is not named click fraud


def test_window_type_from_the_botnet_flows_of_the_window():
    def wf(cls_beh):
        return pd.DataFrame({"cls": [c for c, _ in cls_beh], "beh": [b for _, b in cls_beh]})
    S, I = ctu13.SHAPE_BIT, ctu13.IDENTITY_BIT
    assert ctu13.window_type(wf([("From-Botnet", S), ("From-Botnet", 0)])) == "Shape"
    assert ctu13.window_type(wf([("From-Botnet", I)])) == "Identity"
    assert ctu13.window_type(wf([("From-Botnet", S), ("From-Botnet", I)])) == "Mixed"
    assert ctu13.window_type(wf([("From-Botnet", S | I)])) == "Mixed"
    assert ctu13.window_type(wf([("From-Botnet", 0), ("From-Normal", S)])) == "Unnamed"   # only botnet flows count


def test_typed_cells_use_type_positives_and_all_normals(built):
    c, out = built
    tables = e4.load_tables(out, [1, 2, 3, 4])
    df, typed = e4.run_fold(c, tables, 1, typed=True)
    assert set(typed.wtype) <= set(e4.WTYPES) and len(typed) > 0
    t1 = tables[1]
    n_norm = int((t1.label == 0).sum())
    for T, g in typed.groupby("wtype"):
        assert (g.n_positive == int((t1.wtype == T).sum())).all() and (g.n_normal == n_norm).all()
    assert set(t1.loc[t1.label == 1, "wtype"]) <= set(e4.WTYPES) and (t1.loc[t1.label == 0, "wtype"] == "").all()
    for col in ("false_alarms_per_day", "runtime_per_window_ms", "val_alarm_rate"):   # same fold rows, so same values
        assert typed[col].notna().all()


def test_f3_halves_for_e3x():
    from src.feature_sets import FeatureSet
    rng = np.random.default_rng(0)
    train, test = rng.normal(size=(20, 60, 4)), rng.normal(size=(4, 60, 4))
    full = FeatureSet("F3").fit(train).transform(test)
    h0, h1 = FeatureSet("F3-H0").fit(train).transform(test), FeatureSet("F3-H1").fit(train).transform(test)
    assert h0.shape == (4, 16) and h1.shape == (4, 16)
    assert np.allclose(np.hstack([h0, h1]), full)
    assert FeatureSet("F1+F2+F3-H0").fit(train).names()[-16:] == FeatureSet("F3").fit(train).names()[:16]


def test_summary_by_window_type_on_toy_results():
    sys.path.insert(0, "scripts")
    pytest.importorskip("summarize_smd")
    import summarize_e4 as s4
    rng = np.random.default_rng(1)
    rows = [{"machine": f"scenario-{s}", "scenario": s, "seed": sd, "feature_set": fs, "wtype": t,
             "pr_auc": 0.5 + rng.normal(0, .01), "roc_auc": .8, "recall_at_thr": .5, "false_alarms_per_day": 1.,
             "runtime_per_window_ms": 1., "val_alarm_rate": .01, "n_positive": 12, "n_normal": 100}
            for s in range(1, 9) for sd in (0, 1) for fs in e4.SETS for t in ("Shape", "Mixed")]
    v = s4.by_wtype(pd.DataFrame(rows), 0)
    assert set(v.wtype) == {"Shape", "Mixed"} and (v.n_scenarios == 8).all()
    assert (v.pooled_positive_windows == 96).all()                          # 8 scenarios x 12
