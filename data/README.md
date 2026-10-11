# Data (not committed)

`data/*` is gitignored; only this README is tracked.

| Dataset | Source | Files | SHA-256 |
|---|---|---|---|
| SMD | github.com/NetManAIOps/OmniAnomaly (`ServerMachineDataset/`) @ `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4` | `data/SMD/`: 113 files (below) | per-file list below; digest of the list: `4c5c17f61affcc489fce356ef28c04f306f585595fbb3f6cc17d3f7769d880a0` |
| CTU-13 | Official Stratosphere Mega backup, https://mega.nz/#F!vdRmBA6D!yMZXx74nnu8GjhdwSF54Sw (linked from stratosphereips.org/datasets-ctu13) | `data/CTU-13/<capture>/`: 12 `.binetflow.xz` + 13 READMEs (below). **Scenario 10 (capture 51) has no `.binetflow`** | per-file list below; digest of the list: `13c898fcedb14718946b0499739c0fc03a1b1cdb45538d75d173c5183d504cec` |

## SMD

- **Source:** https://github.com/NetManAIOps/OmniAnomaly, folder `ServerMachineDataset/`, commit `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4` (2021-02-12). Repository licence: MIT (`data/SMD/LICENSE`, © 2021 NetManAIOps-SMD).
- **Downloaded:** 2026-11-19, via a sparse git checkout of that commit; copied unchanged into `data/SMD/`.
- **Layout:** `train/`, `test/`, `test_label/`, `interpretation_label/`, one `machine-<g>-<i>.txt` per machine (28 machines), plus `LICENSE`.
  - `train/`, `test/`: comma-separated, one row per time step, 38 columns (metrics already scaled to [0, 1] by the providers). No timestamps. No NaNs.
  - `test_label/`: one 0/1 label per test row.
  - `interpretation_label/`: which metrics contribute to each anomaly segment (not used by the prereg).
- **Check:** totals match the OmniAnomaly README table: train 708,405, test 708,420, anomaly rate 4.16%.

### Sampling interval

**Not stated in the OmniAnomaly README**, which says only that SMD is "a new 5-week-long dataset". We use **1 minute** as the working value, from Section 2.1 of a secondary source, which gives no citation for it: *Revisiting OmniAnomaly for Anomaly Detection: performance metrics and comparison with PCA-based models*, arXiv:2603.18985. **Consistency check:** at 1 minute per row, the full series (train plus test) lasts 32.9 days (about 4.7 weeks) for the 23.7k-row machines and 39.9 days (about 5.7 weeks) for the 28.7k-row machines. Both are roughly "5 weeks". This value only affects the false-alarms-per-day metric (prereg §6). It is logged under prereg §10 item 2 and does not change any rule.

### Per-machine lengths

| Machine | Train rows | Test rows | Anomalous test points | Anomaly % |
|---|---:|---:|---:|---:|
| machine-1-1 | 28479 | 28479 | 2694 | 9.46 |
| machine-1-2 | 23694 | 23694 | 542 | 2.29 |
| machine-1-3 | 23702 | 23703 | 817 | 3.45 |
| machine-1-4 | 23706 | 23707 | 720 | 3.04 |
| machine-1-5 | 23705 | 23706 | 100 | 0.42 |
| machine-1-6 | 23688 | 23689 | 3708 | 15.65 |
| machine-1-7 | 23697 | 23697 | 2398 | 10.12 |
| machine-1-8 | 23698 | 23699 | 763 | 3.22 |
| machine-2-1 | 23693 | 23694 | 1170 | 4.94 |
| machine-2-2 | 23699 | 23700 | 2833 | 11.95 |
| machine-2-3 | 23688 | 23689 | 269 | 1.14 |
| machine-2-4 | 23689 | 23689 | 1694 | 7.15 |
| machine-2-5 | 23688 | 23689 | 980 | 4.14 |
| machine-2-6 | 28743 | 28743 | 424 | 1.48 |
| machine-2-7 | 23696 | 23696 | 417 | 1.76 |
| machine-2-8 | 23702 | 23703 | 161 | 0.68 |
| machine-2-9 | 28722 | 28722 | 1755 | 6.11 |
| machine-3-1 | 28700 | 28700 | 308 | 1.07 |
| machine-3-2 | 23702 | 23703 | 1109 | 4.68 |
| machine-3-3 | 23703 | 23703 | 632 | 2.67 |
| machine-3-4 | 23687 | 23687 | 977 | 4.12 |
| machine-3-5 | 23690 | 23691 | 426 | 1.80 |
| machine-3-6 | 28726 | 28726 | 1194 | 4.16 |
| machine-3-7 | 28705 | 28705 | 434 | 1.51 |
| machine-3-8 | 28703 | 28704 | 1371 | 4.78 |
| machine-3-9 | 28713 | 28713 | 303 | 1.06 |
| machine-3-10 | 23692 | 23693 | 1047 | 4.42 |
| machine-3-11 | 28695 | 28696 | 198 | 0.69 |
| **Total** | **708405** | **708420** | **29444** | **4.16** |

Train and test lengths differ by at most 1 row per machine. Two length groups: about 23.7k rows (19 machines) and about 28.5–28.7k rows (9 machines).

### SHA-256 checksums

Paths are relative to `data/SMD/`. To verify: `cd data/SMD && sha256sum -c` with the list below.

```
ea2b75dacb9cd140d81962203b18706e8e85d70c87dcced69746a37ba133fa48  ./interpretation_label/machine-1-1.txt
761b4b339a14d4781f34a3cd03fb3d983b01e3bece3f638fa21bb887128d3a11  ./interpretation_label/machine-1-2.txt
dbe9970c43e64445b89211f26eb13f0eccb4da869deaea298218d7c094672c8f  ./interpretation_label/machine-1-3.txt
25978fb0b28261b327a6a50e21ec0c7a232b6a2ad796b1a023b7bfc3d3c19c8a  ./interpretation_label/machine-1-4.txt
c3ab9437aeb07763bba39298dc99905a9365f27f441411d979a9bb66e26a53e9  ./interpretation_label/machine-1-5.txt
2cc68339b99fbf9ec0891c2a081c2d05ee04087a766d6200572fc74e99867f8f  ./interpretation_label/machine-1-6.txt
3a272c538dce66358e9d8f7c18f62f14c652ffb1e52dfc18464cd092860a363b  ./interpretation_label/machine-1-7.txt
0908b5d09296f8379a70c055fdf7e5ebd5f19419e886eab7fca0c766512187ae  ./interpretation_label/machine-1-8.txt
fe4e3a67039d46045b9a3b18fd4dd6138f5c228447fcefb0c5f4beb1f46ede37  ./interpretation_label/machine-2-1.txt
a27f31facae7a9948ab6ba4ce039271df95763c53c1de7f7db699c317db7b233  ./interpretation_label/machine-2-2.txt
2ae064468235de25955439978c56d8fb217bf4682eab73636834e9d69086a777  ./interpretation_label/machine-2-3.txt
b559b694af8d02d23714ea74c526e85f951e1b57b0cb9d59dcd677da84947327  ./interpretation_label/machine-2-4.txt
7a83c2c8058f60c76c9b57cdeafb857594fc92c09f77fe704d2910f554b1cd87  ./interpretation_label/machine-2-5.txt
31f40cffdcc5ff99f00eb605fcfc30fe1fe80b4a59aef6567110cbd3d27e9074  ./interpretation_label/machine-2-6.txt
61b6ffc489cc22c3c08c514c59dde1bc6a0f3991cc964aff0626555e151241fe  ./interpretation_label/machine-2-7.txt
9d8a9f766645f5cca4c9dd48cbeea15db5defe872c0e6eb002e575a9c381c844  ./interpretation_label/machine-2-8.txt
253ca7dab14449f92c7ee08a270b0627808c5dc715ff39ded1cc492ef62fce52  ./interpretation_label/machine-2-9.txt
cd1f1df76a273f3df12bc5389850a0376a361256e0a9ebd6014e08e216661eb7  ./interpretation_label/machine-3-10.txt
ebfd4cd3802c4813de72003d5cee76905c9912d5c0f81403708881e187d27083  ./interpretation_label/machine-3-11.txt
52f67a2be8538138b614850392a232b3f96c9eda987eba41aa2c31b534fd3e2f  ./interpretation_label/machine-3-1.txt
3b76375af29ae9bd873f0746650d293cdb043235953c7b3ca19d893971a44c69  ./interpretation_label/machine-3-2.txt
25da8eda22a12fedc4a3009351d6bbbfa45ed5d5f7b7aaec458840bae2378239  ./interpretation_label/machine-3-3.txt
0a43f6c9ca99345a95ad7de615f275c72692c905e9e92061cdf28ce488a390fc  ./interpretation_label/machine-3-4.txt
baf251f03b57247b4f15f77e48f5238beb16a429c6211d599a825c69c23f6d96  ./interpretation_label/machine-3-5.txt
add62013ecd204670dedf2f0b4c4960ccbcbecf77088dd7712e4714c6311d7f6  ./interpretation_label/machine-3-6.txt
55534eeb66ba496b494c8bac83fe6102d45308bfd2411d152a633d003a20b0c2  ./interpretation_label/machine-3-7.txt
7dda43ca582298295fc84348fd577f7f8b0c6ab67be94bf1e8f9adcf1f26992b  ./interpretation_label/machine-3-8.txt
8f1e4987c967faefb63ad7ebd9d6f01101507e3d908336e4e86bde33b1cb8acb  ./interpretation_label/machine-3-9.txt
c9bef1aab5be4ae6e0735a9f5974de043b8a1894bd8a2a7a163e36027b8895df  ./LICENSE
cddf33441ae480c6074e173c897d4715eb893f24a7078d3d3e1b79b200c7cd19  ./test_label/machine-1-1.txt
c2dbb3e4f89c9eb48f30fd1ddae48b80e7ae7ad7ca9abf2ffd71b6557ef2bb20  ./test_label/machine-1-2.txt
755d57179625682172ae85f9483749cff576ff3282d0af294271a40ab6ea7b28  ./test_label/machine-1-3.txt
a470a3f7b66ec3d080fc5ee3d9565e0f3bfcfa16a1650259f771736327fa9a8b  ./test_label/machine-1-4.txt
e7e2e611e12b5e9e2558ed955ad271f1f408fbdd7fbbfdfc72eb745ade567815  ./test_label/machine-1-5.txt
8359ffa67b5421b0b9967ccf396a58c7b3bf52fa24c3f95e6b9df6486f1d6365  ./test_label/machine-1-6.txt
24b3ce00bf2c56b571775a2b28d4e50ed27d050d77a031fa16c11ef3e51809b8  ./test_label/machine-1-7.txt
8019476032c4c5b4b80127cfb9e74ee7d57fbbd558aa15097518713e16463a47  ./test_label/machine-1-8.txt
416dd20a447f38b8d9394382293ed44f509c1c3f99c4aa2fbff179498626dbdc  ./test_label/machine-2-1.txt
b09e71b3aafe71d9ee73a4058b30efcf67dc80df98d3b7b7a8ec019ea3b769c4  ./test_label/machine-2-2.txt
5768ce5280cb5ce4bee6d8bb3c7105819574d5e82f7d517d61495efa9d4f566e  ./test_label/machine-2-3.txt
c26a6dabfd2ab4aec34c40919100e13e52e17ad0e6b702dd038f582b2c633b01  ./test_label/machine-2-4.txt
9966e4092bece4e0ef60ef54d85396c49c22da64334aad82a9fc2f3842d2ef15  ./test_label/machine-2-5.txt
6fe3dd209542dc6a20879618c40ef6f5d55ce60f2da2902c311d9f0fff3c1ee0  ./test_label/machine-2-6.txt
d9d7aceca750277dd28883f7b5b34da3653652ea9c96de78baa84daeebc4a9b7  ./test_label/machine-2-7.txt
22fbe94e06e56b6035fcc10383bac90d84ff4604f9063009a3940747f92d8105  ./test_label/machine-2-8.txt
eab9425847aa5c7c63485047a49b1bfc9cf44ed88a0d1be84e2fd0a16d0ba895  ./test_label/machine-2-9.txt
d2b87c9c1d1950f0b29eaca0edf4706b917984799f6d9b5c1472435832fe3abd  ./test_label/machine-3-10.txt
ce306777bd66e9ea81de726018e440d49afcc98415cb0db7d99367dddf398f1b  ./test_label/machine-3-11.txt
549348a9c63ed922e85157885bcb21289a8c9f5af94b50147a21cb7b928220ec  ./test_label/machine-3-1.txt
1bbf4223cc6083f718f1d6022e2a48d48af44b08cdfa80048aa51af34d5111c3  ./test_label/machine-3-2.txt
4e2c657f613759f11f167c5540d047723d41d8a3c76467fa4daafcc8a7112035  ./test_label/machine-3-3.txt
5c1f09b85c53d1936eff1e044c5f7413b07f10fd641934b4aba9b2669485b7be  ./test_label/machine-3-4.txt
8a17a0dff8cc0062124997d665a2055cc1c08daffffb9a63ccfd21f7d10bea8f  ./test_label/machine-3-5.txt
2bef9db1d998172c61008f593a7537377179ef4f24963001203c1e48345e9b61  ./test_label/machine-3-6.txt
80b3a19dfd0af2d800c2388474543bdcb723ba582b0f4a986d68f920b757c73c  ./test_label/machine-3-7.txt
dc834a5b5e9a194f5367423c29410afcb622e0e47d7d3362979dd280de713220  ./test_label/machine-3-8.txt
bdaf308affb4653e3e5826177e7e27c883294d2fa8cf48364850ef5a4dac2282  ./test_label/machine-3-9.txt
bdb801d8d52ce6b0c4a20c310edf31b5cee9bc82eb84fb38e7d4a19309b16ddb  ./test/machine-1-1.txt
f409991af7c049d3f286a8cf89dc2fa99114bea8753f6c3aea4bd24cb03eb015  ./test/machine-1-2.txt
f3c134d45ffa7aed66587dba893111396c8c7ea3b6fa7f76895955b26b1a75ef  ./test/machine-1-3.txt
98ade00e57eac63e8f81a506a8e848803b58590835ab2b3cf53d1f0459900bd3  ./test/machine-1-4.txt
98b8979f21b4b68c7308f38dd093b9bff4275814578365103c692f28f60f61c5  ./test/machine-1-5.txt
0797fe4bfd25ef13c830955cb9f2a36b4c9d5da265b8e91702408b6b048267f4  ./test/machine-1-6.txt
785a3f542ef6ac171a3af55c534f9ca598d53fa0371010172c72e33af4588d16  ./test/machine-1-7.txt
b8784c3b4169f8a0a2ed9f47b00070d72ed214dae0307553b8cfdb80c712d779  ./test/machine-1-8.txt
2d103661799271be958227907950a1c76803356442d60a4070671cc0abbefb20  ./test/machine-2-1.txt
1cb0dd0a4b8738b184c62bb07700e6800e029f7a76a715fe0ee147c603b2f2c8  ./test/machine-2-2.txt
eaa24c78ff438eec2f847c09eb4de6ab8dbe8a0bb8d77b52ac4ecdee5bd21a7d  ./test/machine-2-3.txt
85ba3101b9ab6a7684cd897b2d4c78e85caa6f5ef1c4f0e153479f4fca01cf38  ./test/machine-2-4.txt
842588acfb9299735a785757fcadd98cc7f69b6a511d28e54e1af05c5e95137b  ./test/machine-2-5.txt
c895144b7c7fb1ff960f506132d5aeefebce4325efad908d071fb7df5b3b1cf1  ./test/machine-2-6.txt
58c3dfd109437ccbedab9d36608d32874c2ae5473484c5941f85349baf32cef8  ./test/machine-2-7.txt
846207955c546f0c3ea57aa7032b8e3c207ad67632a0605c8273f182059d1a72  ./test/machine-2-8.txt
8d93ccd1a6606de3f665cb2e905d206b60ad2b2a58f47e9561b9bb67a2cd4f28  ./test/machine-2-9.txt
febaf197d4a8864f3a70a5e5c0926d6dab8f1b515cf0d90d31ee81a22d80392f  ./test/machine-3-10.txt
c62f52e65d1b892ef8c556b38aa5c1d3504cf0cdd97a94e7df08985b922d7af0  ./test/machine-3-11.txt
b110b204ba4919250351eba7aa253d73ab54af0cf18e272b9b284de5fa7df88d  ./test/machine-3-1.txt
5f83ded5fbfb9b682030bdd7dd82eb53a77f9caa3773ddcc7e38db73c25db46e  ./test/machine-3-2.txt
ecc3140ce24869a44ebac9b03ca372fb1a6eb7e5bde96946f1a350283d4b152a  ./test/machine-3-3.txt
07e1b43e20b7d8300fcd4b4f45ea47ea8c851f3dbd7d1fe971139c32f039357e  ./test/machine-3-4.txt
85dbb13b3076981ad917f42081b71a1440d8b2aafd9bee626f0d1cacff43fcd6  ./test/machine-3-5.txt
04f746a4dde41adbfe5727c6ff77839d1b0b21d6d65bdb0253557e0391c5d44c  ./test/machine-3-6.txt
a4588c1f39905d03bd24fce65052e7b175fba336d081843325ef780c9ed8cdad  ./test/machine-3-7.txt
fa79d5dad3a703fbc13faf0eab26d9466b74789f2bb0d22776f2adaa4236f085  ./test/machine-3-8.txt
5a8968c269a8a3562e54235aa05344b262f44829c2bae04cf9f5d364f388310d  ./test/machine-3-9.txt
ecebd2933ccdc1126e2abbfb538762dbde16410cebc77257a7b96e1e1fbcf753  ./train/machine-1-1.txt
7f6f63ec34ce616c87b4c30d66fd52ad1b21149634ae3ff7bb1827a76d6ef79a  ./train/machine-1-2.txt
a4ef16a49d921e0ba9497ce2690d3833eb4a85f9a8a88392de4c3fde42e413ac  ./train/machine-1-3.txt
2ec56c43f91684aa4751a6b72d359669266712dd69bdc7efda3dc60c0ade1156  ./train/machine-1-4.txt
69061055f43413863434cb9c0424c14b13a7a0817056a8fd9439e53dc365fc0b  ./train/machine-1-5.txt
27e1765c04c2a5d08f39d98e7979e7764886fc733c36382fe081157a5ac0b9d8  ./train/machine-1-6.txt
dddb9f770bdb8b1b952263e0cc4f000b6f8ec612941dd3eb7a303994f257edf8  ./train/machine-1-7.txt
3b46e9754ec06bebd0bf3fe68dbe28b2a4b67298e795c9501d97f42083e337fc  ./train/machine-1-8.txt
d6eb13ff74a537cf33686319fffdd4cdcb394862570a3acd519b4c37af97cd2e  ./train/machine-2-1.txt
e2e0f52994c8c884f9552c6289e42268d9d988ba7c7abadcc0d8dcd8ba993c30  ./train/machine-2-2.txt
d2b0fc37d08cb727438598a1b09ced94f71f7792a9b4d057242e2ce8f2580e13  ./train/machine-2-3.txt
077a5c18649b9d260d8dcbd4cfbf10b16cb83719a52013410cf3b5dc6a802fa6  ./train/machine-2-4.txt
940cdf20a8345fdeeb8043efd7ce9f7506fe0e7f5f89f8b50edf19b61b304136  ./train/machine-2-5.txt
dc30adaef939060aebea3836c691e3d1bb47da593fa43553acbc52c6825dc8ae  ./train/machine-2-6.txt
32e9d88b18bf1b377a2bd0a6d5de2a766f36c58781e76dbe2491b0347557dec2  ./train/machine-2-7.txt
01ae9289566e3518cf738518a25edb0812f889e8ed1dada057f4693ac5dda23c  ./train/machine-2-8.txt
465db893338b6f14985dd30ac2ff20a319246bb8584cd0a9a4aaacf52cbc2f1f  ./train/machine-2-9.txt
8f8258b3a85fe8aa5badb05e66b0cdd3405be5f180f0d418bc24c30503207fd2  ./train/machine-3-10.txt
a2e531cf2bf9b8d833a0b6e7c42a596b1d3a64578a1c140537add46156bde4e9  ./train/machine-3-11.txt
c1186ac085809c0590467dd42f4f12180012ebd994c1e615250dbed2a3588111  ./train/machine-3-1.txt
7cabed178422f7772425c6cabc3c20a1f7dedc3ab2a950f90a9edfcef3b5b3e3  ./train/machine-3-2.txt
6bbad775c6ef6319abba08aaa6faa2befb2cd570f9db85b80413ae6fdf1047a4  ./train/machine-3-3.txt
a27a88c5d96c3384e07776dd2ced9956f7ea78b1070b87786352bc6024129d72  ./train/machine-3-4.txt
7eb4f157d5aace0f5758c8035a4cc90445fc0d3ae59c57b1bf00d7e62c363bba  ./train/machine-3-5.txt
88b8164dccfe67704e83c0919aa40bf1d4b8d682aff0b947d4cdd1ec4f900b84  ./train/machine-3-6.txt
5eb1f16e62c30eb498e968b2c6da3e8f60c8367e264f771245a970362d7d8574  ./train/machine-3-7.txt
414ba622ae70118948a3940009ce4e90580b494ea07f9e03893c26e102edd855  ./train/machine-3-8.txt
7896b57b65da1d48ee992ba128fb914ae7e6d2bef8ea4ed8df039657fc017468  ./train/machine-3-9.txt
```

## CTU-13

- **Source:** the official Stratosphere backup on Mega, https://mega.nz/#F!vdRmBA6D!yMZXx74nnu8GjhdwSF54Sw, which is linked from https://www.stratosphereips.org/datasets-ctu13. That page notes: "keep in mind that the main site is the authoritative copy". The main site (`mcfp.felk.cvut.cz`, archive `CTU-13-Dataset.tar.bz2`) refused connections from this laptop on 2026-11-19 (two `wget -c` attempts timed out).
- **Downloaded:** 2026-11-19 by Anirudh, per scenario folder. Only the labelled bidirectional NetFlow files and the READMEs; no pcaps.
- **Format:** `capture*.binetflow.xz`, xz-compressed CSV (Argus bidirectional NetFlow). Kept compressed; pandas reads `.xz` directly. All 12 archives pass `xz -t`.
- **Folder = capture number.** Scenario = capture − 41, per the scenario list on the Stratosphere page:

| Scenario | Folder | File | Flows |
|---:|---|---|---:|
| 1 | 42 | capture20110810.binetflow.xz | 2,824,636 |
| 2 | 43 | capture20110811.binetflow.xz | 1,808,122 |
| 3 | 44 | capture20110812.binetflow.xz | 4,710,638 |
| 4 | 45 | capture20110815.binetflow.xz | 1,121,076 |
| 5 | 46 | capture20110815-2.binetflow.xz | 129,832 |
| 6 | 47 | capture20110816.binetflow.xz | 558,919 |
| 7 | 48 | capture20110816-2.binetflow.xz | 114,077 |
| 8 | 49 | capture20110816-3.binetflow.xz | 2,954,230 |
| 9 | 50 | capture20110817.binetflow.xz | 2,087,508 |
| 10 | 51 | **missing** (README only; expected `capture20110818.binetflow`) | — |
| 11 | 52 | capture20110818-2.binetflow.xz | 107,251 |
| 12 | 53 | capture20110819.binetflow.xz | 325,471 |
| 13 | 54 | capture20110815-3.binetflow.xz | 1,925,149 |

Label facts (prereg §10 item 1) are in `docs/ctu13_facts.md`.

### CTU-13 SHA-256 checksums

Paths are relative to `data/CTU-13/`. To verify: `cd data/CTU-13 && sha256sum -c` with the list below.

```
ab71abcaa575b10e329606057bd37d46a53cda2ed27ec45f3395d34dfbd4380a  42/capture20110810.binetflow.xz
63bead68db3724972731a134fb117d71c17584b27bfa220e8bc1f1d435aa3f94  42/README.html
178ff0275a26bf543a58ac49d3dd84804f524057a825416acaf94e5810d70543  43/capture20110811.binetflow.xz
f7f3c037ea39fa716ad388b330e2a48645bbf209762098eaa0c35f2138a0e612  43/README
6b855d9e2a736cdc13f44d044ec115d131597ddb24d2d85ccce685c4906854c5  44/capture20110812.binetflow.xz
8d4d391f2cefbf12e9fd5aa97159dc6047a14e6a4a83b88ece27c44a23c18fb4  44/README-44
3ddff4a0e0208ca2e6c98aca443d534ad3737f5976b7c6aaddd25122fdae6966  45/capture20110815.binetflow.xz
1826dd7913cffdab5d1907e784ec1af9040aec1dcc6f7f094ec3d32b5561eaf5  45/README-45
77ee55723502d78b4673293718818dd4474476e75656227f80616f6033d0e02a  46/capture20110815-2.binetflow.xz
52b68b8f537463dec56d766652b053ac1122d653062698a609a765e7099ad8cd  46/README-46
743b02e18a66facefad46dc107a88180a36ab53476e929327a12a9f4eeda787c  47/capture20110816.binetflow.xz
f002a281128bfb1dd25762c4740ff6e02013f16c1bb82db5e6ddfe46a37b8c7a  47/README-47
785c7abbd9169eb454e12edf68f5431081d3514607e5fa1086731861d285795a  48/capture20110816-2.binetflow.xz
10019db9572422993a6153c6a999c74a45c1268e180681284cdf6c74ca96ae65  48/README-48
6f558061b76dee6f859489d1a5c1cd6a3a5d93682b325c587bb9f626eab930bc  49/capture20110816-3.binetflow.xz
5fc9b33dc9f33286792b1d20a7e095c7f96347ec478e7358bfd53ee8db091a4a  49/README-49
57cdad839df2697052f2f8152ac05fa06ca420f1ee0a847a3d27be06a2afc321  50/capture20110817.binetflow.xz
aca7f0a37ad1c403bf61ac5540f5ec7650b10c741c41d08bb8a631c822751218  50/README-50
8d53b13d21c9495909b1c2257e50a5bfe228b3105a8b4027385b53194ded0931  51/README-51
e925db754b09a1490c609a624a316ec40542ff47aca4fc58853df9959fe7319a  52/capture20110818-2.binetflow.xz
dbd8baca23479b4b88b43860805fd5914b0a0dc5dd3e7081648642703b9a7976  52/README-52
11069acc42352ecc33e11b38ac92bfd43c6decc3652a41de29e0547d8970c754  53/capture20110819.binetflow.xz
ab38c7a72341fef075f16825fff9ae7a39d873e6343452b73216b521e35516cf  53/README-53
979ed61c91c1f52c1086401b0de89428b8d4153048f74abe5f2b7204dbff201e  54/capture20110815-3.binetflow.xz
93cd63f011c79eaef16970d628256c9486013e2e4566c2430fa81e446a00ec9c  54/README-54
```
