"""v24 = enlarged names dev set run (user approved 2026-11-08 11:16, "you can add even more people"): 105-cell / 49-person EVAL_NAMES (v5) with 50 held-out IdNrs; driver runs seeds 1,2,3 as subprocesses on the arm-B manifest minus held-out (216 cells); per-seed timing_result_seedN.json. v23 = multi-seed noise sizing (user approved 2026-11-08 09:45): driver runs seeds 1,2,3 as subprocesses on the unchanged 306-cell arm-B manifest + lowlr recipe; per-seed timing_result_seedN.json kept, checkpoints deleted between seeds. Question: does a v18-style step-250 names-CER dip (0.8894) occur by chance across seeds? v22 = v21 + recursive kurrent-code resolution (train_manifest_v4, eval_names) + embedded names-eval fallback + boot assert. v21 = v20 with mount-proof dataset resolution (Kaggle nested /kaggle/input/datasets/<user>/<slug> mounts broke PAGE_DIRS and the kurrent-code lookup) + embedded eval500 fallback. v20 = v18/v19 pipeline with the arm-B enlarged target set: 306 cells (301 merged
v15 + 5 second-hand right-page date cells: g10/38752, g10/3664, g07/38779,
g07/38653, g07/38765). Rule extension approved by user 2026-11-08 08:35
(corroborated time words + parent-name cells at medium; applied: all time words
here uncorroborated -> excluded; weekdays only where calendar-verified + legible).
Recipe: lowlr LR 1e-6, REPEAT 4, from base, MAX_STEPS 500, EVAL_EVERY 125,
init-seeded names-CER selection (v18 trajectory: peak @250, overfit @375+).
Manifest embedded below (merged_armb_manifest.json); held-out asserts unchanged.
"""
import os
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

import os as _dos, sys as _dsys, subprocess as _dsp, shutil as _dsh
if "KURRENT_SEED" not in _dos.environ:
    # v23 multi-seed driver: 3 fresh seeds as sequential subprocesses (fresh process = clean GPU memory)
    for _s in (1, 2, 3):
        print(f"===== DRIVER seed {_s} start =====", flush=True)
        _r = _dsp.run([_dsys.executable, "/kaggle/src/script.py"], env=dict(_dos.environ, KURRENT_SEED=str(_s)))
        if _dos.path.exists("/kaggle/working/timing_result.json"):
            _dsh.copy("/kaggle/working/timing_result.json", f"/kaggle/working/timing_result_seed{_s}.json")
            _dos.remove("/kaggle/working/timing_result.json")
        for _d in ("ckpt", "checkpoint-final", "checkpoint-best"):
            _dsh.rmtree(f"/kaggle/working/{_d}", ignore_errors=True)
        print(f"===== DRIVER seed {_s} exit={_r.returncode} =====", flush=True)
    _dsys.exit(0)
SEED = int(_dos.environ["KURRENT_SEED"])
print(f"STAGE seed={SEED}", flush=True)

T0 = time.time()
import glob as _glob

SMOKE = os.environ.get("KURRENT_SMOKE") == "1"
CODE = Path(os.environ.get("KURRENT_CODE_DIR", "/kaggle/input/kurrent-code"))
DATA_ROOT = Path(os.environ.get("KURRENT_DATA_ROOT", "/kaggle/input"))

_hits = _glob.glob(str(DATA_ROOT / "**/eval_manifest.jsonl"), recursive=True)
print("STAGE manifest-search:", _hits, flush=True)
DATA = Path(_hits[0]).parent if _hits else DATA_ROOT / "kurrent-train-v1"
# pages are split across multiple dataset mounts (Kaggle file-count limit):
# v21: mounts can nest (e.g. /kaggle/input/datasets/<user>/<slug>) -> resolve recursively
def _find_ds_dirs(slug_glob):
    out = []
    for base in (DATA_ROOT, DATA_ROOT / "datasets"):
        out += sorted(base.glob(slug_glob))
        out += sorted(base.glob(f"*/{slug_glob}"))
    seen, res = set(), []
    for d in out:
        if d.is_dir() and str(d) not in seen:
            seen.add(str(d))
            res.append(d)
    return res
PAGE_DIRS = _find_ds_dirs("kurrent-train-v1*")
PAGE_DIRS += _find_ds_dirs("kurrent-batch2")  # v14: also carries g01_22.jpg (not in the v1 mounts)
_code_dirs = _find_ds_dirs("kurrent-code")
if _code_dirs:
    CODE = _code_dirs[0]
print("STAGE code-dir:", str(CODE), flush=True)
PAGE_ROOT = {}
for _d in PAGE_DIRS:
    for _f in _d.iterdir():
        if _f.suffix == ".jpg":
            PAGE_ROOT[_f.name] = _f
print("STAGE page-roots:", [str(d) for d in PAGE_DIRS], "pages:", len(PAGE_ROOT), flush=True)

# v21: eval500 manifest resolves recursively; embedded blob is the fallback if the code mount moved
_EVAL500_HITS = _glob.glob(str(DATA_ROOT / "**/eval_mm006.jsonl"), recursive=True)
print("STAGE eval500-search:", _EVAL500_HITS, flush=True)
_EVAL500_BLOB = '''{"file": "01501.jpg", "bbox": [302, 648, 754, 81], "text": "Gemeinden Deputierte, welche we¬", "project": "MM_1_006"}
{"file": "01469.jpg", "bbox": [363, 583, 737, 87], "text": "sel noch vorhanden sind, diesen Waß¬", "project": "MM_1_006"}
{"file": "01562.jpg", "bbox": [301, 785, 724, 90], "text": "dienliche Weise zur Kenntniß der", "project": "MM_1_006"}
{"file": "01550.jpg", "bbox": [320, 2722, 743, 100], "text": "in Genehmigung des Commißional-A¬", "project": "MM_1_006"}
{"file": "01542.jpg", "bbox": [295, 2789, 697, 109], "text": "des zu erbrechenden beydsei¬", "project": "MM_1_006"}
{"file": "01511.jpg", "bbox": [323, 2348, 677, 91], "text": "zu 12. jähriger Kettenstrafe ver¬", "project": "MM_1_006"}
{"file": "01497.jpg", "bbox": [346, 2720, 729, 90], "text": "inger angefangenen Huldigungs¬", "project": "MM_1_006"}
{"file": "01663.jpg", "bbox": [339, 1439, 676, 107], "text": "gen die Huldigung bis jezt noch", "project": "MM_1_006"}
{"file": "01492.jpg", "bbox": [296, 572, 743, 101], "text": "bezirk Horgen nach Wädenschweil sich", "project": "MM_1_006"}
{"file": "01617.jpg", "bbox": [315, 2430, 715, 97], "text": "in Folge Ratserkanntnuß vom 27ten", "project": "MM_1_006"}
{"file": "01471.jpg", "bbox": [367, 2139, 660, 73], "text": "Das von Burgermeister und", "project": "MM_1_006"}
{"file": "01470.jpg", "bbox": [338, 2904, 644, 94], "text": "lung jener Kirchenguts-Rech¬", "project": "MM_1_006"}
{"file": "01494.jpg", "bbox": [348, 1969, 709, 85], "text": "verhaftet befindet, auch Vieharzt", "project": "MM_1_006"}
{"file": "01541.jpg", "bbox": [333, 770, 742, 85], "text": "den Kleinen Rath, hievon dem Herrn", "project": "MM_1_006"}
{"file": "01546.jpg", "bbox": [324, 1753, 754, 100], "text": "haften Bemühungen, deren vollen Werth", "project": "MM_1_006"}
{"file": "01648.jpg", "bbox": [338, 1437, 194, 76], "text": "versehen.", "project": "MM_1_006"}
{"file": "01469.jpg", "bbox": [352, 2166, 726, 101], "text": "Holz zu unterstüzen, dieser allge¬", "project": "MM_1_006"}
{"file": "01533.jpg", "bbox": [295, 2333, 724, 91], "text": "ohne Ansehen seiner Heimath, er", "project": "MM_1_006"}
{"file": "01663.jpg", "bbox": [337, 1025, 708, 104], "text": "beßtens verdankt, ihre Verfügun¬", "project": "MM_1_006"}
{"file": "01616.jpg", "bbox": [335, 2806, 478, 71], "text": "eingegeben werden.", "project": "MM_1_006"}
{"file": "01541.jpg", "bbox": [356, 2445, 713, 92], "text": "als sie allein betreffend, zurükge¬", "project": "MM_1_006"}
{"file": "01627.jpg", "bbox": [331, 1936, 665, 91], "text": "aufwärts, und auf Rüti. In Rüti", "project": "MM_1_006"}
{"file": "01563.jpg", "bbox": [322, 1273, 729, 90], "text": "hießigen Cantons insbesondere,", "project": "MM_1_006"}
{"file": "01462.jpg", "bbox": [283, 1531, 747, 95], "text": "meinde Oberurdorf gethane Ansuchen,", "project": "MM_1_006"}
{"file": "01519.jpg", "bbox": [380, 639, 750, 89], "text": "meldten Stadtbezirksgericht mittelst", "project": "MM_1_006"}
{"file": "01618.jpg", "bbox": [334, 408, 745, 100], "text": "in der 2ten Abtheilung des Bezirks", "project": "MM_1_006"}
{"file": "01586.jpg", "bbox": [307, 2104, 734, 93], "text": "Unterstatthalteren abschriftlich mit¬", "project": "MM_1_006"}
{"file": "01563.jpg", "bbox": [323, 920, 730, 101], "text": "die gewesenen gemeineidsge¬", "project": "MM_1_006"}
{"file": "01517.jpg", "bbox": [338, 2039, 667, 93], "text": "ächtlich, und vermittelst Hinter¬", "project": "MM_1_006"}
{"file": "01539.jpg", "bbox": [324, 2542, 276, 84], "text": "behalten soll.", "project": "MM_1_006"}
{"file": "01585.jpg", "bbox": [332, 1234, 664, 81], "text": "werfen, nachzufolgen, und in den", "project": "MM_1_006"}
{"file": "01497.jpg", "bbox": [338, 2425, 738, 92], "text": "schienen, auch überhaupt zu seiner", "project": "MM_1_006"}
{"file": "01494.jpg", "bbox": [351, 1023, 734, 91], "text": "wie eine beträchtliche Anzahl min¬", "project": "MM_1_006"}
{"file": "01602.jpg", "bbox": [352, 1213, 695, 84], "text": "Herrn Landammanns, derselben getha¬", "project": "MM_1_006"}
{"file": "01495.jpg", "bbox": [320, 2345, 655, 84], "text": "markt dem, mit Baumwollen", "project": "MM_1_006"}
{"file": "01593.jpg", "bbox": [339, 1582, 653, 98], "text": "re durch einen vom hießigen", "project": "MM_1_006"}
{"file": "01587.jpg", "bbox": [286, 2931, 685, 86], "text": "nebst einigen Danikern und", "project": "MM_1_006"}
{"file": "01558.jpg", "bbox": [330, 689, 737, 87], "text": "Stütze für das Fortkommen seines", "project": "MM_1_006"}
{"file": "01476.jpg", "bbox": [367, 649, 202, 83], "text": "bringen.", "project": "MM_1_006"}
{"file": "01631.jpg", "bbox": [268, 1379, 717, 90], "text": "nen Mittwoch den 28ten Merz beym", "project": "MM_1_006"}
{"file": "01659.jpg", "bbox": [345, 3104, 714, 91], "text": "balprozeß über die, am 19ten dieß", "project": "MM_1_006"}
{"file": "01506.jpg", "bbox": [353, 1129, 649, 88], "text": "wiederholte ihnen obige Ver¬", "project": "MM_1_006"}
{"file": "01600.jpg", "bbox": [348, 2401, 664, 83], "text": "der Mediation, die anderen der", "project": "MM_1_006"}
{"file": "01489.jpg", "bbox": [318, 406, 743, 91], "text": "geschäft, und den eingelegten beiden", "project": "MM_1_006"}
{"file": "01665.jpg", "bbox": [320, 2638, 667, 90], "text": "daß endlich der Hebräer Wolf", "project": "MM_1_006"}
{"file": "01569.jpg", "bbox": [320, 1069, 707, 103], "text": "gestern Morgen gegen 11. Uhr wie¬", "project": "MM_1_006"}
{"file": "01594.jpg", "bbox": [287, 1698, 737, 83], "text": "öffentlich bekannt, und zweifeln nicht,", "project": "MM_1_006"}
{"file": "01531.jpg", "bbox": [330, 1169, 683, 119], "text": "Zurükstellung der Originalien", "project": "MM_1_006"}
{"file": "01485.jpg", "bbox": [253, 1855, 766, 107], "text": "gen der Mehrzahl unter Euch gewe¬", "project": "MM_1_006"}
{"file": "01477.jpg", "bbox": [265, 406, 774, 91], "text": "Von dieser Ernennung wird sowohl", "project": "MM_1_006"}
{"file": "01544.jpg", "bbox": [322, 2328, 686, 85], "text": "Excellenz, des Herrn Landammanns", "project": "MM_1_006"}
{"file": "01567.jpg", "bbox": [328, 2854, 688, 89], "text": "am 16ten dieß, und zu Birmenstorf", "project": "MM_1_006"}
{"file": "01489.jpg", "bbox": [329, 1309, 710, 100], "text": "en Meyer ab Wildsperg Auskunft,", "project": "MM_1_006"}
{"file": "01546.jpg", "bbox": [328, 1934, 745, 91], "text": "ganz von dem gewünschten Erfolg beglei¬", "project": "MM_1_006"}
{"file": "01497.jpg", "bbox": [326, 1227, 754, 101], "text": "gegeben habe, wurde beschloßen, den", "project": "MM_1_006"}
{"file": "01602.jpg", "bbox": [353, 1375, 687, 101], "text": "Jakob Willi von Horgen eine Prä¬", "project": "MM_1_006"}
{"file": "01563.jpg", "bbox": [335, 1059, 701, 96], "text": "auf gedachter Extraconferenz,", "project": "MM_1_006"}
{"file": "01629.jpg", "bbox": [322, 697, 751, 100], "text": "6o. Von dem Gang der Untersuchung", "project": "MM_1_006"}
{"file": "01595.jpg", "bbox": [315, 2296, 711, 87], "text": "hann Melchior Christen aus der", "project": "MM_1_006"}
{"file": "01520.jpg", "bbox": [327, 968, 716, 91], "text": "den Handverkauf wegen folgender", "project": "MM_1_006"}
{"file": "01597.jpg", "bbox": [356, 2413, 735, 109], "text": "der hießigen Regierung gefällig seyn", "project": "MM_1_006"}
{"file": "01592.jpg", "bbox": [328, 568, 713, 87], "text": "und eigentlichen Sinn der den Zehen¬", "project": "MM_1_006"}
{"file": "01537.jpg", "bbox": [321, 2341, 659, 88], "text": "räthe aber spätestens bis könfti¬", "project": "MM_1_006"}
{"file": "01559.jpg", "bbox": [311, 389, 721, 91], "text": "wahreidgenößischer Freude über", "project": "MM_1_006"}
{"file": "01486.jpg", "bbox": [323, 1089, 741, 93], "text": "stüzung des von dem Herrn Un¬", "project": "MM_1_006"}
{"file": "01523.jpg", "bbox": [332, 1257, 648, 75], "text": "wendung derselben in andern", "project": "MM_1_006"}
{"file": "01659.jpg", "bbox": [313, 1074, 742, 71], "text": "richten, ob und wie sie die Procla¬", "project": "MM_1_006"}
{"file": "01550.jpg", "bbox": [317, 2610, 749, 96], "text": "vom 3ten hujus, diesem Auftrag ein", "project": "MM_1_006"}
{"file": "01520.jpg", "bbox": [325, 1702, 737, 89], "text": "2o. Die von dem Käufer in seinen Ko¬", "project": "MM_1_006"}
{"file": "01632.jpg", "bbox": [309, 1449, 745, 87], "text": "Christian Woern von Schönaich, aus dem", "project": "MM_1_006"}
{"file": "01601.jpg", "bbox": [298, 647, 231, 97], "text": "bemühe.", "project": "MM_1_006"}
{"file": "01560.jpg", "bbox": [350, 879, 645, 85], "text": "und Polizey-Commißion, als dem", "project": "MM_1_006"}
{"file": "01541.jpg", "bbox": [348, 1661, 721, 79], "text": "dießfälligen Verfügungen, und in", "project": "MM_1_006"}
{"file": "01580.jpg", "bbox": [311, 2074, 709, 97], "text": "man gestehen, daß auch die, rük¬", "project": "MM_1_006"}
{"file": "01480.jpg", "bbox": [351, 2064, 734, 96], "text": "als Ersaz für den Scheuchzerischen", "project": "MM_1_006"}
{"file": "01545.jpg", "bbox": [320, 780, 710, 83], "text": "mes Auge zu bestellen, und die", "project": "MM_1_006"}
{"file": "01471.jpg", "bbox": [370, 2433, 664, 100], "text": "zey ein verdächtiger Mensch,", "project": "MM_1_006"}
{"file": "01577.jpg", "bbox": [314, 1822, 730, 79], "text": "Alle ansehnlichen Leuthe mußten sich", "project": "MM_1_006"}
{"file": "01610.jpg", "bbox": [362, 769, 685, 79], "text": "chen Standescommißion, als Admini¬", "project": "MM_1_006"}
{"file": "01559.jpg", "bbox": [313, 1170, 738, 97], "text": "langens, daß ihm der Gemeindrath", "project": "MM_1_006"}
{"file": "01484.jpg", "bbox": [295, 1280, 722, 97], "text": "Herrn Oberst Husers von Glarus ste¬", "project": "MM_1_006"}
{"file": "01538.jpg", "bbox": [318, 1051, 701, 78], "text": "greßes beyfügt, um vielleicht desto", "project": "MM_1_006"}
{"file": "01577.jpg", "bbox": [314, 1121, 686, 82], "text": "te, einen vortheilhaften Posten zu", "project": "MM_1_006"}
{"file": "01538.jpg", "bbox": [299, 2627, 753, 83], "text": "terbringen, ob und was in dieser An¬", "project": "MM_1_006"}
{"file": "01646.jpg", "bbox": [355, 547, 660, 98], "text": "den Collatur-Ansprüche zum", "project": "MM_1_006"}
{"file": "01608.jpg", "bbox": [286, 850, 726, 81], "text": "zeigen, worauf dieser aber nur", "project": "MM_1_006"}
{"file": "01631.jpg", "bbox": [423, 803, 323, 87], "text": "Signalement.", "project": "MM_1_006"}
{"file": "01512.jpg", "bbox": [303, 2934, 712, 93], "text": "im Zeughaus zu Brugg vorgefun¬", "project": "MM_1_006"}
{"file": "01558.jpg", "bbox": [337, 1050, 740, 90], "text": "selbe aber noch für 4. Jahre in sei¬", "project": "MM_1_006"}
{"file": "01511.jpg", "bbox": [313, 2410, 711, 79], "text": "urtheilten Jacob Wüst von Kloten,", "project": "MM_1_006"}
{"file": "01551.jpg", "bbox": [341, 1626, 698, 95], "text": "sein möchte, den verfallenen Grund¬", "project": "MM_1_006"}
{"file": "01660.jpg", "bbox": [320, 3057, 740, 101], "text": "gen Auftrag in allen Fällen [nach", "project": "MM_1_006"}
{"file": "01557.jpg", "bbox": [325, 1761, 704, 104], "text": "und die über die beschränkte Zahl", "project": "MM_1_006"}
{"file": "01620.jpg", "bbox": [326, 531, 707, 89], "text": "wähnten falschen Vierbazen-Stüke von", "project": "MM_1_006"}
{"file": "01609.jpg", "bbox": [341, 1718, 674, 96], "text": "aber – die Chefs der bewaffneten", "project": "MM_1_006"}
{"file": "01594.jpg", "bbox": [291, 1998, 733, 93], "text": "gebliebene große Mehrzahl unserer", "project": "MM_1_006"}
{"file": "01541.jpg", "bbox": [352, 1392, 728, 67], "text": "wurde, thätig gewesen, und wie er", "project": "MM_1_006"}
{"file": "01511.jpg", "bbox": [321, 1097, 693, 113], "text": "zu weiterer Berathung und Ver¬", "project": "MM_1_006"}
{"file": "01649.jpg", "bbox": [305, 2981, 736, 84], "text": "Commißion des Innern mit der Einla¬", "project": "MM_1_006"}
{"file": "01643.jpg", "bbox": [321, 2871, 706, 86], "text": "ner bedenklicher Verflechtungen,", "project": "MM_1_006"}
{"file": "01493.jpg", "bbox": [267, 1845, 760, 96], "text": "hat heute seine Sizungen auf dem", "project": "MM_1_006"}
{"file": "01477.jpg", "bbox": [282, 1673, 758, 73], "text": "der Commißion des Innern als der Lobl.", "project": "MM_1_006"}
{"file": "01500.jpg", "bbox": [311, 1722, 683, 85], "text": "Herren Bezirks- und Unterstatt¬", "project": "MM_1_006"}
{"file": "01516.jpg", "bbox": [323, 2106, 735, 91], "text": "auf dieses Fundament hin, einen reif¬", "project": "MM_1_006"}
{"file": "01519.jpg", "bbox": [395, 1063, 730, 82], "text": "nats dem Kleinen Rathe unterstü¬", "project": "MM_1_006"}
{"file": "01617.jpg", "bbox": [303, 2665, 757, 75], "text": "len im Canton Thurgau nun wirklich", "project": "MM_1_006"}
{"file": "01483.jpg", "bbox": [332, 1450, 657, 83], "text": "nem oder ein paar Individuen", "project": "MM_1_006"}
{"file": "01604.jpg", "bbox": [344, 548, 749, 94], "text": "fortdauerndes Verbleiben habe;", "project": "MM_1_006"}
{"file": "01602.jpg", "bbox": [347, 2852, 706, 100], "text": "klärt, dem- oder denjenigen hinge¬", "project": "MM_1_006"}
{"file": "01634.jpg", "bbox": [344, 1667, 610, 84], "text": "erachtende Weise für ihre", "project": "MM_1_006"}
{"file": "01657.jpg", "bbox": [344, 1277, 638, 101], "text": "dem Anfang der Revolution", "project": "MM_1_006"}
{"file": "01553.jpg", "bbox": [315, 1076, 760, 92], "text": "6o. Was die Geschreyungen und an¬", "project": "MM_1_006"}
{"file": "01666.jpg", "bbox": [339, 1206, 697, 107], "text": "der Sachen Auskunft geben zu laß¬", "project": "MM_1_006"}
{"file": "01464.jpg", "bbox": [319, 738, 741, 93], "text": "anzuweisen, um vermittelst die¬", "project": "MM_1_006"}
{"file": "01502.jpg", "bbox": [333, 1071, 664, 85], "text": "gesezt worden sind, – so bleibt es", "project": "MM_1_006"}
{"file": "01660.jpg", "bbox": [340, 1055, 711, 85], "text": "des neuerwählten Herrn Pfarrers", "project": "MM_1_006"}
{"file": "01558.jpg", "bbox": [329, 2997, 739, 89], "text": "und Klarheit der bestehenden Zehen¬", "project": "MM_1_006"}
{"file": "01586.jpg", "bbox": [310, 2029, 738, 105], "text": "den sämtlichen Herren Bezirks[-] und", "project": "MM_1_006"}
{"file": "01501.jpg", "bbox": [301, 815, 748, 100], "text": "entstandenen schmerzhaften Gefüh¬", "project": "MM_1_006"}
{"file": "01569.jpg", "bbox": [334, 1379, 645, 93], "text": "rischer Form nach Stäfa mar¬", "project": "MM_1_006"}
{"file": "01622.jpg", "bbox": [335, 1434, 740, 96], "text": "angebotten worden, von der hießi¬", "project": "MM_1_006"}
{"file": "01518.jpg", "bbox": [330, 1906, 709, 100], "text": "klärt und die dortige Regierung", "project": "MM_1_006"}
{"file": "01629.jpg", "bbox": [325, 1010, 708, 74], "text": "Nach Anhörung des unterm 2ten", "project": "MM_1_006"}
{"file": "01461.jpg", "bbox": [327, 1010, 739, 94], "text": "ligen, nach richterlicher Ermäßi¬", "project": "MM_1_006"}
{"file": "01557.jpg", "bbox": [320, 2400, 684, 92], "text": "der gedachten Huldigungen und", "project": "MM_1_006"}
{"file": "01646.jpg", "bbox": [349, 1928, 681, 107], "text": "sie dießfalls gänzlich auszuwei¬", "project": "MM_1_006"}
{"file": "01526.jpg", "bbox": [350, 745, 696, 86], "text": "vergreiffen wollet? Dann, Einwoh¬", "project": "MM_1_006"}
{"file": "01649.jpg", "bbox": [335, 986, 735, 92], "text": "chem Ende hin die Justiz- und Polizei", "project": "MM_1_006"}
{"file": "01499.jpg", "bbox": [320, 1498, 739, 83], "text": "lichen Polizey-Anstalten, nach Maaß¬", "project": "MM_1_006"}
{"file": "01571.jpg", "bbox": [325, 687, 740, 86], "text": "chen Dank zu bezeugen, und von der,", "project": "MM_1_006"}
{"file": "01533.jpg", "bbox": [297, 2405, 686, 99], "text": "mag verfolgt, angehalten oder", "project": "MM_1_006"}
{"file": "01599.jpg", "bbox": [353, 539, 689, 92], "text": "Verbrechens beschuldigt ist. Jedem", "project": "MM_1_006"}
{"file": "01519.jpg", "bbox": [402, 2593, 743, 76], "text": "bäude von 33’ Länge und 31’ Breite", "project": "MM_1_006"}
{"file": "01661.jpg", "bbox": [350, 784, 656, 94], "text": "sazung eingestellt, zumahlen", "project": "MM_1_006"}
{"file": "01657.jpg", "bbox": [340, 2633, 674, 97], "text": "reich aus dem Grund verhin¬", "project": "MM_1_006"}
{"file": "01460.jpg", "bbox": [339, 923, 741, 91], "text": "3o. Von diesen Verfügungen wird", "project": "MM_1_006"}
{"file": "01580.jpg", "bbox": [310, 1876, 750, 107], "text": "des freyen Verkehrs keiner Anwen¬", "project": "MM_1_006"}
{"file": "01642.jpg", "bbox": [331, 462, 677, 106], "text": "gerichteten Zuschrift darüber", "project": "MM_1_006"}
{"file": "01467.jpg", "bbox": [335, 457, 695, 84], "text": "mit der gewißenhaftetsten Rük¬", "project": "MM_1_006"}
{"file": "01501.jpg", "bbox": [303, 1127, 731, 85], "text": "möglichsten Beruhigung verbschei¬", "project": "MM_1_006"}
{"file": "01594.jpg", "bbox": [297, 3110, 715, 106], "text": "des Herren Landammanns der Schweiz in", "project": "MM_1_006"}
{"file": "01574.jpg", "bbox": [288, 2014, 754, 106], "text": "an den helvetischen Schulden zu über¬", "project": "MM_1_006"}
{"file": "01548.jpg", "bbox": [319, 2955, 699, 101], "text": "gewesenen Verwaltungskam¬", "project": "MM_1_006"}
{"file": "01481.jpg", "bbox": [332, 1470, 676, 96], "text": "Allervorderst sey auf dem Plan", "project": "MM_1_006"}
{"file": "01549.jpg", "bbox": [298, 1540, 694, 89], "text": "ihren Antrag zu hinterbringen.", "project": "MM_1_006"}
{"file": "01488.jpg", "bbox": [356, 3079, 685, 79], "text": "telbaren und absoluten Colla-", "project": "MM_1_006"}
{"file": "01491.jpg", "bbox": [328, 1919, 719, 112], "text": "lichster Beschleunigung in der", "project": "MM_1_006"}
{"file": "01641.jpg", "bbox": [316, 770, 694, 103], "text": "beseze, und so in der fortgehenden", "project": "MM_1_006"}
{"file": "01485.jpg", "bbox": [263, 1483, 723, 88], "text": "jenigen, welche selbst ihre Zeit, ihre", "project": "MM_1_006"}
{"file": "01658.jpg", "bbox": [357, 2351, 720, 103], "text": "Eglisau jährlich entrichteten Zinses", "project": "MM_1_006"}
{"file": "01506.jpg", "bbox": [348, 2051, 690, 73], "text": "rung, und eine, – wenn schon nicht", "project": "MM_1_006"}
{"file": "01507.jpg", "bbox": [313, 1938, 679, 100], "text": "rung der Geseze abzielende Bitte", "project": "MM_1_006"}
{"file": "01637.jpg", "bbox": [343, 933, 710, 90], "text": "auf von der Stadtbürgerschaft von", "project": "MM_1_006"}
{"file": "01664.jpg", "bbox": [332, 2987, 653, 102], "text": "fortzusezen, dieselben auf dem", "project": "MM_1_006"}
{"file": "01521.jpg", "bbox": [227, 529, 755, 91], "text": "Ufer zu sezen. Die Mannschaft ward", "project": "MM_1_006"}
{"file": "01558.jpg", "bbox": [337, 1242, 679, 84], "text": "seines Gemeindraths gestellt.", "project": "MM_1_006"}
{"file": "01656.jpg", "bbox": [369, 2484, 687, 89], "text": "ten des Herrn Salomon Arters", "project": "MM_1_006"}
{"file": "01618.jpg", "bbox": [333, 848, 727, 92], "text": "In der Kirche zu Thalwyl wurde", "project": "MM_1_006"}
{"file": "01538.jpg", "bbox": [307, 1857, 203, 80], "text": "terstüzen.", "project": "MM_1_006"}
{"file": "01661.jpg", "bbox": [339, 658, 682, 82], "text": "Gutachtens über die Instructi¬", "project": "MM_1_006"}
{"file": "01534.jpg", "bbox": [351, 1910, 640, 92], "text": "re Achtung, der beßte Dank der", "project": "MM_1_006"}
{"file": "01576.jpg", "bbox": [368, 971, 715, 96], "text": "3o. Soll diese Publikation, wenn sie", "project": "MM_1_006"}
{"file": "01609.jpg", "bbox": [337, 1390, 696, 96], "text": "Reue gefoltert, vor Euch darnieder", "project": "MM_1_006"}
{"file": "01598.jpg", "bbox": [345, 2812, 706, 83], "text": "zu Handen der Einwohner unsers", "project": "MM_1_006"}
{"file": "01623.jpg", "bbox": [352, 2008, 665, 108], "text": "den. Zugleich bezeuget der Klei¬", "project": "MM_1_006"}
{"file": "01652.jpg", "bbox": [322, 2987, 723, 98], "text": "be diese Notariatskanzley mit aller", "project": "MM_1_006"}
{"file": "01628.jpg", "bbox": [350, 1456, 673, 70], "text": "mannsstelle, wird die Militaircom¬", "project": "MM_1_006"}
{"file": "01504.jpg", "bbox": [341, 2668, 664, 111], "text": "mit herzlicher Freude erfüllt,", "project": "MM_1_006"}
{"file": "01551.jpg", "bbox": [331, 2946, 691, 84], "text": "laßes für dieselben, in ihrer bis¬", "project": "MM_1_006"}
{"file": "01543.jpg", "bbox": [330, 1584, 659, 101], "text": "derselbe [in Folge erhaltenen", "project": "MM_1_006"}
{"file": "01483.jpg", "bbox": [346, 1978, 665, 100], "text": "tags der Actus durch aufrühreri¬", "project": "MM_1_006"}
{"file": "01585.jpg", "bbox": [331, 2638, 667, 84], "text": "1o. Die, von der außerordentli¬", "project": "MM_1_006"}
{"file": "01467.jpg", "bbox": [327, 2040, 680, 113], "text": "dig sind, auch in dieser wichti¬", "project": "MM_1_006"}
{"file": "01666.jpg", "bbox": [340, 1841, 314, 81], "text": "gen genehmigt.", "project": "MM_1_006"}
{"file": "01545.jpg", "bbox": [324, 1961, 697, 87], "text": "einen geringen Nuzen abwer¬", "project": "MM_1_006"}
{"file": "01541.jpg", "bbox": [352, 2264, 744, 83], "text": "ren Landammann der Schweiz zur Er¬", "project": "MM_1_006"}
{"file": "01462.jpg", "bbox": [279, 2146, 743, 100], "text": "der allerdings begründet befunde¬", "project": "MM_1_006"}
{"file": "01486.jpg", "bbox": [306, 643, 741, 100], "text": "dem Weg anderweitige Erkundi¬", "project": "MM_1_006"}
{"file": "01481.jpg", "bbox": [333, 2409, 655, 107], "text": "dort die Thur einen ganz gera¬", "project": "MM_1_006"}
{"file": "01545.jpg", "bbox": [330, 659, 689, 94], "text": "Polizey-Commißion mitgetheilt,", "project": "MM_1_006"}
{"file": "01484.jpg", "bbox": [295, 2460, 728, 101], "text": "welche sie nur als eine Folge des", "project": "MM_1_006"}
{"file": "01471.jpg", "bbox": [372, 1776, 677, 95], "text": "sen benannten Gemeinden die", "project": "MM_1_006"}
{"file": "01582.jpg", "bbox": [318, 2827, 650, 112], "text": "theilt, damit die geistlichen", "project": "MM_1_006"}
{"file": "01486.jpg", "bbox": [317, 446, 712, 101], "text": "an die Argauische Regierung zu", "project": "MM_1_006"}
{"file": "01651.jpg", "bbox": [324, 2306, 751, 94], "text": "gegeben, und Sr. Excellenz dem Herrn", "project": "MM_1_006"}
{"file": "01548.jpg", "bbox": [339, 1385, 706, 76], "text": "ven enthaltene Bericht erstattet.", "project": "MM_1_006"}
{"file": "01563.jpg", "bbox": [322, 1621, 747, 80], "text": "ven enthaltene, verbindliche Dank¬", "project": "MM_1_006"}
{"file": "01640.jpg", "bbox": [265, 2977, 234, 83], "text": "bescheinen.", "project": "MM_1_006"}
{"file": "01539.jpg", "bbox": [332, 1500, 675, 103], "text": "das Seebett so vorrüken zu dür¬", "project": "MM_1_006"}
{"file": "01661.jpg", "bbox": [337, 607, 655, 94], "text": "on nächstens vorzulegenden", "project": "MM_1_006"}
{"file": "01509.jpg", "bbox": [315, 413, 737, 86], "text": "Vaterland seyen Zeugen unsers fe¬", "project": "MM_1_006"}
{"file": "01636.jpg", "bbox": [329, 832, 126, 68], "text": "könne.", "project": "MM_1_006"}
{"file": "01550.jpg", "bbox": [315, 851, 735, 101], "text": "Frauenfeld ein anderes Project ei¬", "project": "MM_1_006"}
{"file": "01636.jpg", "bbox": [337, 1232, 731, 95], "text": "von Regenstorf vom 17ten pti. ergiebt,", "project": "MM_1_006"}
{"file": "01612.jpg", "bbox": [331, 1269, 732, 77], "text": "an den meistbietenden verkauft,", "project": "MM_1_006"}
{"file": "01530.jpg", "bbox": [343, 1965, 691, 89], "text": "ligen, geschikten und eifrigen, mit", "project": "MM_1_006"}
{"file": "01494.jpg", "bbox": [341, 2578, 722, 94], "text": "So wie nun die weitere Entwik¬", "project": "MM_1_006"}
{"file": "01495.jpg", "bbox": [306, 2621, 672, 91], "text": "gestohlen, zwischen seinen Depo¬", "project": "MM_1_006"}
{"file": "01621.jpg", "bbox": [364, 399, 715, 110], "text": "Das von dem Obergericht den 3t[e¬", "project": "MM_1_006"}
{"file": "01591.jpg", "bbox": [323, 2274, 724, 98], "text": "dieses Stük Land 12. Jahre lang Brach", "project": "MM_1_006"}
{"file": "01618.jpg", "bbox": [336, 1228, 756, 93], "text": "der großen Mehrheit der vorhande¬", "project": "MM_1_006"}
{"file": "01613.jpg", "bbox": [328, 2559, 669, 97], "text": "gen erlittenen Verlust an 993.", "project": "MM_1_006"}
{"file": "01634.jpg", "bbox": [333, 630, 668, 102], "text": "schäft beendige, wobey das Be¬", "project": "MM_1_006"}
{"file": "01479.jpg", "bbox": [333, 2937, 724, 89], "text": "gen, sondern auch den Bericht über", "project": "MM_1_006"}
{"file": "01496.jpg", "bbox": [342, 1467, 661, 106], "text": "deßhalben nöthige Publikation", "project": "MM_1_006"}
{"file": "01482.jpg", "bbox": [327, 1453, 675, 106], "text": "ber er sich schon vergeblich an die", "project": "MM_1_006"}
{"file": "01610.jpg", "bbox": [351, 2211, 631, 74], "text": "Aprill erwähnt sind, und des", "project": "MM_1_006"}
{"file": "01586.jpg", "bbox": [307, 1156, 752, 98], "text": "ten Memorials gegen die, von dem", "project": "MM_1_006"}
{"file": "01500.jpg", "bbox": [305, 1391, 695, 94], "text": "nem angemeßenen Anhang von", "project": "MM_1_006"}
{"file": "01552.jpg", "bbox": [374, 981, 636, 83], "text": "ken könnte, mit der vollstän¬", "project": "MM_1_006"}
{"file": "01531.jpg", "bbox": [325, 514, 703, 100], "text": "der Justiz- und Polzey-Commißi¬", "project": "MM_1_006"}
{"file": "01530.jpg", "bbox": [349, 1791, 666, 83], "text": "gungsregister im Original zu¬", "project": "MM_1_006"}
{"file": "01660.jpg", "bbox": [328, 369, 734, 87], "text": "zösischen Truppen, seit ihrem lezten", "project": "MM_1_006"}
{"file": "01627.jpg", "bbox": [335, 1399, 681, 96], "text": "wo er mit derselben nach Stäfa", "project": "MM_1_006"}
{"file": "01511.jpg", "bbox": [321, 2991, 658, 101], "text": "Entweichungsanschlägen thätig", "project": "MM_1_006"}
{"file": "01527.jpg", "bbox": [281, 2626, 713, 99], "text": "Stimmung derselben war ungleich.", "project": "MM_1_006"}
{"file": "01563.jpg", "bbox": [320, 1688, 720, 92], "text": "sagungsschreiben abzuerlaßen.", "project": "MM_1_006"}
{"file": "01632.jpg", "bbox": [308, 1782, 711, 87], "text": "kauft, und auf Martini 1804. die", "project": "MM_1_006"}
{"file": "01552.jpg", "bbox": [370, 2159, 656, 95], "text": "die Herstellung der Ruhe und", "project": "MM_1_006"}
{"file": "01487.jpg", "bbox": [353, 2429, 669, 97], "text": "und durch denselben an diese", "project": "MM_1_006"}
{"file": "01625.jpg", "bbox": [321, 1359, 691, 98], "text": "könnenden Personen vor sich beschei¬", "project": "MM_1_006"}
{"file": "01665.jpg", "bbox": [313, 826, 668, 89], "text": "hard Ziegler vom National-S¬", "project": "MM_1_006"}
{"file": "01496.jpg", "bbox": [341, 997, 697, 99], "text": "toren des erstern aufgefordert wer¬", "project": "MM_1_006"}
{"file": "01478.jpg", "bbox": [337, 2272, 647, 91], "text": "betrachten sind, und wo einzig", "project": "MM_1_006"}
{"file": "01661.jpg", "bbox": [333, 1955, 663, 83], "text": "monat auf der in Rheinau ab¬", "project": "MM_1_006"}
{"file": "01465.jpg", "bbox": [292, 1256, 761, 96], "text": "ertheilt habe, und demnach dem Her¬", "project": "MM_1_006"}
{"file": "01494.jpg", "bbox": [345, 1540, 689, 100], "text": "Flügel, dann alt Gerichtspräsident", "project": "MM_1_006"}
{"file": "01547.jpg", "bbox": [304, 2427, 650, 93], "text": "für die hießigen Ansprüche auf", "project": "MM_1_006"}
{"file": "01521.jpg", "bbox": [237, 1670, 750, 93], "text": "Über die Umstände, welche die Hin¬", "project": "MM_1_006"}
{"file": "01612.jpg", "bbox": [328, 601, 747, 97], "text": "Ao. 1800. ohne Zins benuzen zu können,", "project": "MM_1_006"}
{"file": "01641.jpg", "bbox": [319, 485, 703, 58], "text": "lezten Thurgauischen Proposition[en] dahin gehen, daß", "project": "MM_1_006"}
{"file": "01639.jpg", "bbox": [280, 1283, 708, 89], "text": "gen zu erscheinen aufgefordert", "project": "MM_1_006"}
{"file": "01539.jpg", "bbox": [334, 1058, 667, 92], "text": "2o. Da alt Kirchenpfleger Hs: Ru¬", "project": "MM_1_006"}
{"file": "01610.jpg", "bbox": [358, 655, 685, 84], "text": "Kleinen Rathe oder auf deßen An¬", "project": "MM_1_006"}
{"file": "01481.jpg", "bbox": [341, 2155, 646, 89], "text": "sich begreiffe, und unter dem", "project": "MM_1_006"}
{"file": "01520.jpg", "bbox": [334, 2790, 745, 100], "text": "brunnen zu dem yeinigen in seinen", "project": "MM_1_006"}
{"file": "01600.jpg", "bbox": [348, 2903, 683, 96], "text": "oder persönliche Mißhandlung", "project": "MM_1_006"}
{"file": "01460.jpg", "bbox": [336, 2561, 740, 87], "text": "fenden Gemeinds- oder jeder ande¬", "project": "MM_1_006"}
{"file": "01606.jpg", "bbox": [51, 1052, 249, 54], "text": "Obersten Ziegler zum", "project": "MM_1_006"}
{"file": "01558.jpg", "bbox": [336, 1422, 740, 95], "text": "wohl als dem Herrn Unterstatt¬", "project": "MM_1_006"}
{"file": "01629.jpg", "bbox": [320, 2302, 748, 95], "text": "b. Daß die hießigen HHerren Depu¬", "project": "MM_1_006"}
{"file": "01566.jpg", "bbox": [327, 1009, 740, 114], "text": "Achtung zuwiderlaufe] abzulehnen", "project": "MM_1_006"}
{"file": "01618.jpg", "bbox": [329, 745, 736, 87], "text": "wo Kilchberg und Adlischwyl ver¬", "project": "MM_1_006"}
{"file": "01641.jpg", "bbox": [328, 1381, 656, 75], "text": "betreffenden Pfarreren gegeben", "project": "MM_1_006"}
{"file": "01517.jpg", "bbox": [345, 1432, 637, 83], "text": "mißion, im Original, als auch", "project": "MM_1_006"}
{"file": "01530.jpg", "bbox": [338, 1477, 696, 94], "text": "gesezlicher Vorschrift und zu gänz¬", "project": "MM_1_006"}
{"file": "01570.jpg", "bbox": [321, 1845, 743, 90], "text": "ter Frik unterm 25ten Merz, in Fol¬", "project": "MM_1_006"}
{"file": "01540.jpg", "bbox": [328, 2345, 169, 63], "text": "laßen.", "project": "MM_1_006"}
{"file": "01481.jpg", "bbox": [330, 2031, 656, 75], "text": "promiß einzig die Linie vom", "project": "MM_1_006"}
{"file": "01662.jpg", "bbox": [297, 948, 742, 86], "text": "ster herkommenden Vorstellungen", "project": "MM_1_006"}
{"file": "01482.jpg", "bbox": [326, 1736, 686, 110], "text": "dieser Forderung und zur gefäl¬", "project": "MM_1_006"}
{"file": "01576.jpg", "bbox": [349, 2870, 688, 85], "text": "Pension von 500. Gulden wirklich", "project": "MM_1_006"}
{"file": "01481.jpg", "bbox": [332, 642, 678, 84], "text": "Octobris a. p. abweichenden Pro¬", "project": "MM_1_006"}
{"file": "01478.jpg", "bbox": [345, 1892, 166, 53], "text": "machen.", "project": "MM_1_006"}
{"file": "01637.jpg", "bbox": [356, 2434, 735, 87], "text": "schloßenen Bemühungen zu Beybe¬", "project": "MM_1_006"}
{"file": "01647.jpg", "bbox": [338, 1555, 679, 109], "text": "zu Handen gestellt werden, auf", "project": "MM_1_006"}
{"file": "01658.jpg", "bbox": [344, 760, 731, 83], "text": "Churhaus Baaden die Abtrettung", "project": "MM_1_006"}
{"file": "01518.jpg", "bbox": [335, 1181, 734, 107], "text": "würde, an deßen Besoldung all¬", "project": "MM_1_006"}
{"file": "01480.jpg", "bbox": [352, 3058, 739, 102], "text": "auch dem Erziehungsrath durch Zu-", "project": "MM_1_006"}
{"file": "01649.jpg", "bbox": [338, 1172, 723, 100], "text": "Schwerzenbach in Gefolge der mit dem¬", "project": "MM_1_006"}
{"file": "01489.jpg", "bbox": [328, 1611, 717, 92], "text": "Kirche unruhige Bewegungen ge¬", "project": "MM_1_006"}
{"file": "01528.jpg", "bbox": [342, 2627, 684, 70], "text": "haltenen Umständen und Angaben,", "project": "MM_1_006"}
{"file": "01485.jpg", "bbox": [262, 823, 760, 115], "text": "aufrichtigste Reue, durch ein ruhiges", "project": "MM_1_006"}
{"file": "01484.jpg", "bbox": [288, 3006, 745, 81], "text": "Auf der rechten Seite des Sees wer¬", "project": "MM_1_006"}
{"file": "01547.jpg", "bbox": [316, 1346, 667, 88], "text": "gen Ehrengesandtschaft auf der Con¬", "project": "MM_1_006"}
{"file": "01611.jpg", "bbox": [329, 646, 757, 77], "text": "Standes mit dem Canton St. Gallen,", "project": "MM_1_006"}
{"file": "01504.jpg", "bbox": [346, 1578, 644, 95], "text": "zirks Horgen, in der Kirche zu", "project": "MM_1_006"}
{"file": "01551.jpg", "bbox": [334, 1197, 692, 79], "text": "ten zu Gunsten mehrerer Gemein¬", "project": "MM_1_006"}
{"file": "01474.jpg", "bbox": [309, 2303, 754, 107], "text": "stungsbegehren des K. K. Beam¬", "project": "MM_1_006"}
{"file": "01490.jpg", "bbox": [355, 805, 700, 105], "text": "gen für das Militarwesen betrof¬", "project": "MM_1_006"}
{"file": "01616.jpg", "bbox": [333, 2491, 665, 84], "text": "nicht einnahmen, wegen dieses", "project": "MM_1_006"}
{"file": "01654.jpg", "bbox": [301, 2637, 640, 118], "text": "erforderlichen Dispositionen zu", "project": "MM_1_006"}
{"file": "01577.jpg", "bbox": [316, 2609, 669, 87], "text": "trauten sich nicht, diese Abrede", "project": "MM_1_006"}
{"file": "01556.jpg", "bbox": [364, 2186, 716, 84], "text": "rath bemerkt wird, daß es aller¬", "project": "MM_1_006"}
{"file": "01535.jpg", "bbox": [322, 2146, 675, 90], "text": "genbliklich auf jeden Ruf der Re¬", "project": "MM_1_006"}
{"file": "01577.jpg", "bbox": [309, 685, 711, 91], "text": "Vereinigung der Einwohner der", "project": "MM_1_006"}
{"file": "01548.jpg", "bbox": [335, 2088, 675, 96], "text": "wißen nöthig habe, ob die Um¬", "project": "MM_1_006"}
{"file": "01558.jpg", "bbox": [348, 1830, 736, 91], "text": "Raths Erkanntnuß hinterbrachten An¬", "project": "MM_1_006"}
{"file": "01608.jpg", "bbox": [279, 967, 750, 91], "text": "de zu antworten wußte. Der Ge¬", "project": "MM_1_006"}
{"file": "01508.jpg", "bbox": [277, 1650, 727, 91], "text": "sammlungen insbesondere halber, be¬", "project": "MM_1_006"}
{"file": "01571.jpg", "bbox": [337, 2140, 725, 82], "text": "Handlung durch das lärmende Gewühl", "project": "MM_1_006"}
{"file": "01630.jpg", "bbox": [288, 1918, 744, 90], "text": "keit des Entschlußes der HHerren De¬", "project": "MM_1_006"}
{"file": "01577.jpg", "bbox": [307, 2445, 737, 84], "text": "gen Sturm zu läuten, und von da", "project": "MM_1_006"}
{"file": "01486.jpg", "bbox": [316, 2285, 736, 99], "text": "nern aber beauftragt seyn, eine", "project": "MM_1_006"}
{"file": "01463.jpg", "bbox": [307, 1441, 738, 90], "text": "sich gezeigten Ruhestörer bereits ein¬", "project": "MM_1_006"}
{"file": "01630.jpg", "bbox": [288, 2935, 717, 92], "text": "dem Kleinen Rathe zu Handen", "project": "MM_1_006"}
{"file": "01496.jpg", "bbox": [345, 3049, 697, 96], "text": "18ten auf den 19ten Merz gewalt¬", "project": "MM_1_006"}
{"file": "01486.jpg", "bbox": [304, 3073, 749, 100], "text": "ken auf angemeßene und gut-", "project": "MM_1_006"}
{"file": "01660.jpg", "bbox": [326, 1691, 701, 101], "text": "Herr Bezirksstatthalter Steiner,", "project": "MM_1_006"}
{"file": "01539.jpg", "bbox": [343, 460, 684, 80], "text": "schiffmeister Bolleters gar keine", "project": "MM_1_006"}
{"file": "01648.jpg", "bbox": [325, 1933, 688, 96], "text": "Rath, daß dem dermaligen Schul¬", "project": "MM_1_006"}
{"file": "01509.jpg", "bbox": [321, 589, 731, 83], "text": "allgemein hergestellt, und die auf¬", "project": "MM_1_006"}
{"file": "01589.jpg", "bbox": [291, 2547, 737, 84], "text": "nung getroffen werden möchte, und", "project": "MM_1_006"}
{"file": "01485.jpg", "bbox": [253, 1119, 760, 95], "text": "Die Gutgesinnten soll das Gefühl ihrer", "project": "MM_1_006"}
{"file": "01550.jpg", "bbox": [319, 2117, 746, 96], "text": "stiger Jahreszeit für die Wuhrung¬", "project": "MM_1_006"}
{"file": "01597.jpg", "bbox": [361, 1773, 710, 81], "text": "verstorbenen Bruders, bescheint, und", "project": "MM_1_006"}
{"file": "01566.jpg", "bbox": [331, 778, 750, 103], "text": "lung, wie sehr eine solche Forderung", "project": "MM_1_006"}
{"file": "01518.jpg", "bbox": [327, 1488, 717, 88], "text": "hießigen Lob[lichen] Stand wiederholt", "project": "MM_1_006"}
{"file": "01623.jpg", "bbox": [352, 2136, 681, 89], "text": "halter Steiner für die auch bey", "project": "MM_1_006"}
{"file": "01662.jpg", "bbox": [303, 2008, 706, 112], "text": "sich die Stimmung verändert zu", "project": "MM_1_006"}
{"file": "01572.jpg", "bbox": [297, 2169, 746, 97], "text": "lung an den Herrn Troitte in Bern", "project": "MM_1_006"}
{"file": "01657.jpg", "bbox": [344, 1090, 702, 96], "text": "Merz, worinn diese beyden Män¬", "project": "MM_1_006"}
{"file": "01462.jpg", "bbox": [278, 2788, 748, 92], "text": "gereyen selbst an die Hand zu geben.", "project": "MM_1_006"}
{"file": "01666.jpg", "bbox": [333, 2472, 694, 96], "text": "über den Ursprung, die Tendenz", "project": "MM_1_006"}
{"file": "01571.jpg", "bbox": [328, 1585, 736, 93], "text": "woraus sich ergiebt, daß die Huldigung", "project": "MM_1_006"}
{"file": "01498.jpg", "bbox": [346, 1022, 669, 88], "text": "kon mitgetheilt habe, um auch", "project": "MM_1_006"}
{"file": "01509.jpg", "bbox": [328, 2315, 734, 102], "text": "Herrn Landammann der Schweiz, über", "project": "MM_1_006"}
{"file": "01502.jpg", "bbox": [341, 2357, 651, 92], "text": "meindskammer provisorisch und", "project": "MM_1_006"}
{"file": "01499.jpg", "bbox": [327, 2076, 736, 80], "text": "Die theils einmüthig, theils mit Mehr¬", "project": "MM_1_006"}
{"file": "01517.jpg", "bbox": [328, 1978, 689, 106], "text": "zwekdienlichen Verfügungen gut¬", "project": "MM_1_006"}
{"file": "01561.jpg", "bbox": [325, 634, 689, 87], "text": "her eine Handänderung der¬", "project": "MM_1_006"}
{"file": "01564.jpg", "bbox": [304, 2463, 713, 84], "text": "sung zugehen zu laßen, sobald ihm", "project": "MM_1_006"}
{"file": "01538.jpg", "bbox": [304, 683, 768, 90], "text": "dem Schweizerischen Handel so nachtheili¬", "project": "MM_1_006"}
{"file": "01587.jpg", "bbox": [287, 1865, 738, 98], "text": "der Eidesleistung mit gehörigem", "project": "MM_1_006"}
{"file": "01535.jpg", "bbox": [319, 1477, 671, 96], "text": "sonders darauf Bedacht zu neh¬", "project": "MM_1_006"}
{"file": "01557.jpg", "bbox": [307, 2968, 707, 82], "text": "dung mit den beiden Seegegen¬", "project": "MM_1_006"}
{"file": "01648.jpg", "bbox": [327, 1483, 691, 99], "text": "Da nun der dermalige Schulmei¬", "project": "MM_1_006"}
{"file": "01642.jpg", "bbox": [346, 393, 697, 102], "text": "renden Herren Burgermeister", "project": "MM_1_006"}
{"file": "01553.jpg", "bbox": [316, 773, 747, 91], "text": "24 ß. 9 Hlr ihr übrig bleiben, so soll", "project": "MM_1_006"}
{"file": "01478.jpg", "bbox": [339, 2504, 679, 95], "text": "des gesezlichen Eides eingekom¬", "project": "MM_1_006"}
{"file": "01494.jpg", "bbox": [339, 580, 758, 96], "text": "des Aufruhrs und der verbrecherischen", "project": "MM_1_006"}
{"file": "01562.jpg", "bbox": [278, 2403, 782, 93], "text": "Kriegsgerichts, geäußerten Bedenken,", "project": "MM_1_006"}
{"file": "01476.jpg", "bbox": [361, 1290, 649, 72], "text": "pti an sämmtliche L. Stände er¬", "project": "MM_1_006"}
{"file": "01461.jpg", "bbox": [322, 1325, 750, 89], "text": "hat dieselbe beschloßen, der Argaui¬", "project": "MM_1_006"}
{"file": "01584.jpg", "bbox": [327, 1126, 729, 96], "text": "neten der Regierung, HHerrn Raths¬", "project": "MM_1_006"}
{"file": "01508.jpg", "bbox": [270, 1582, 755, 93], "text": "sollenden unbefugten Gemeindver¬", "project": "MM_1_006"}
{"file": "01557.jpg", "bbox": [333, 626, 698, 96], "text": "isches Aufsehen zusichert, ist einst¬", "project": "MM_1_006"}
{"file": "01519.jpg", "bbox": [396, 2648, 750, 96], "text": "zu seinem Handwerksgebrauch er¬", "project": "MM_1_006"}
{"file": "01624.jpg", "bbox": [342, 2947, 717, 95], "text": "und zwar der erstere alljährlich", "project": "MM_1_006"}
{"file": "01665.jpg", "bbox": [324, 2221, 667, 102], "text": "Nazional-Schazkammer Strauß", "project": "MM_1_006"}
{"file": "01619.jpg", "bbox": [301, 2647, 705, 98], "text": "auf die von ihm gethane Einfrage,", "project": "MM_1_006"}
{"file": "01463.jpg", "bbox": [308, 1791, 742, 87], "text": "der Kleine Rath, mit tröstlicher Beru¬", "project": "MM_1_006"}
{"file": "01487.jpg", "bbox": [336, 2256, 694, 88], "text": "Kleine Rath erwartet hätte, daß", "project": "MM_1_006"}
{"file": "01515.jpg", "bbox": [296, 1008, 736, 83], "text": "Nach dieser mündlichen Berichts¬", "project": "MM_1_006"}
{"file": "01663.jpg", "bbox": [340, 1631, 689, 86], "text": "2o. Die Rathsmitglieder, welche von", "project": "MM_1_006"}
{"file": "01473.jpg", "bbox": [321, 1114, 702, 96], "text": "und allein um die Leistung des", "project": "MM_1_006"}
{"file": "01597.jpg", "bbox": [360, 1485, 729, 72], "text": "dießeitigen Schreibens vom 11ten Au¬", "project": "MM_1_006"}
{"file": "01666.jpg", "bbox": [342, 450, 702, 88], "text": "lichen Verhöre vornehmen und", "project": "MM_1_006"}
{"file": "01514.jpg", "bbox": [284, 2748, 747, 113], "text": "den, beruhenden Ablehnungen, eben¬", "project": "MM_1_006"}
{"file": "01620.jpg", "bbox": [305, 1798, 696, 94], "text": "merken, daß, ungeachtet sein Schrei¬", "project": "MM_1_006"}
{"file": "01507.jpg", "bbox": [315, 1039, 686, 95], "text": "Bescheide entlaßen, daß die Stan¬", "project": "MM_1_006"}
{"file": "01475.jpg", "bbox": [311, 1800, 693, 112], "text": "nennt, in der vollsten Überzeu¬", "project": "MM_1_006"}
{"file": "01574.jpg", "bbox": [296, 2729, 737, 88], "text": "die Bittschrift des Gemeindrathswai¬", "project": "MM_1_006"}
{"file": "01595.jpg", "bbox": [326, 2093, 732, 79], "text": "wegen verübten Diebstählen, auf", "project": "MM_1_006"}
{"file": "01474.jpg", "bbox": [304, 2453, 730, 102], "text": "billigen, und an den Gränzen", "project": "MM_1_006"}
{"file": "01593.jpg", "bbox": [325, 369, 708, 100], "text": "abgehalten werden kann, welches", "project": "MM_1_006"}
{"file": "01537.jpg", "bbox": [311, 2948, 642, 98], "text": "den aufbreche, um am 23ten", "project": "MM_1_006"}
{"file": "01552.jpg", "bbox": [367, 1907, 669, 100], "text": "Vorstellungen an die aufrühri¬", "project": "MM_1_006"}
{"file": "01497.jpg", "bbox": [331, 2998, 738, 97], "text": "gen besorglich vorfallender straf¬", "project": "MM_1_006"}
{"file": "01591.jpg", "bbox": [331, 1712, 720, 83], "text": "land auf folgende Weise zur Benu¬", "project": "MM_1_006"}
{"file": "01517.jpg", "bbox": [336, 1218, 700, 90], "text": "gemeinntzigsten öffentlichen Zwe¬", "project": "MM_1_006"}
{"file": "01547.jpg", "bbox": [311, 2797, 676, 78], "text": "von 60,000 fl. in eben dem Maaße", "project": "MM_1_006"}
{"file": "01520.jpg", "bbox": [326, 853, 722, 87], "text": "9ten gl. M. mit dem Lehenziegler Hs:", "project": "MM_1_006"}
{"file": "01525.jpg", "bbox": [343, 1894, 672, 102], "text": "dem damaligen Helvetischen Voll¬", "project": "MM_1_006"}
{"file": "01614.jpg", "bbox": [309, 968, 738, 94], "text": "der Zuchthausverwalter Stelle abhangen¬", "project": "MM_1_006"}
{"file": "01469.jpg", "bbox": [365, 400, 742, 82], "text": "ten Steinrad, wo ehemals ein, von", "project": "MM_1_006"}
{"file": "01526.jpg", "bbox": [343, 1357, 366, 72], "text": "Mächte überein.", "project": "MM_1_006"}
{"file": "01583.jpg", "bbox": [344, 2045, 662, 82], "text": "er, zum Fundament des dieß¬", "project": "MM_1_006"}
{"file": "01614.jpg", "bbox": [309, 432, 737, 86], "text": "Zuchthausverwalter Koller einstweilen", "project": "MM_1_006"}
{"file": "01552.jpg", "bbox": [365, 499, 667, 104], "text": "damit größerem Unglük und", "project": "MM_1_006"}
{"file": "01518.jpg", "bbox": [335, 2950, 748, 92], "text": "Besoldung für die gewünschte Sonn¬", "project": "MM_1_006"}
{"file": "01499.jpg", "bbox": [341, 3090, 740, 82], "text": "fügungen dem Kleinen Rath in seinen", "project": "MM_1_006"}
{"file": "01603.jpg", "bbox": [312, 1095, 664, 77], "text": "gefloßenen 6. Mütt Kernen und", "project": "MM_1_006"}
{"file": "01474.jpg", "bbox": [336, 1126, 723, 95], "text": "Zuschrift des Kleinen Raths des L.", "project": "MM_1_006"}
{"file": "01635.jpg", "bbox": [306, 1569, 708, 95], "text": "verweigeret haben, oder wegen", "project": "MM_1_006"}
{"file": "01542.jpg", "bbox": [317, 1872, 665, 110], "text": "die dießfällige Depêsche der", "project": "MM_1_006"}
{"file": "01534.jpg", "bbox": [338, 400, 682, 97], "text": "ter Schwerzenbach, und der von", "project": "MM_1_006"}
{"file": "01631.jpg", "bbox": [268, 2295, 755, 72], "text": "Da der, seit ein Paar Tagen in hießi¬", "project": "MM_1_006"}
{"file": "01590.jpg", "bbox": [360, 452, 680, 97], "text": "mißion bis gegen Ende des Jahrs", "project": "MM_1_006"}
{"file": "01573.jpg", "bbox": [285, 2350, 717, 92], "text": "auf die beiden hießigen Schreiben", "project": "MM_1_006"}
{"file": "01544.jpg", "bbox": [326, 1997, 667, 107], "text": "forderlich sey, um Ruhe und Ord¬", "project": "MM_1_006"}
{"file": "01542.jpg", "bbox": [307, 2443, 702, 81], "text": "communiciert, daß Wohldieselbe", "project": "MM_1_006"}
{"file": "01468.jpg", "bbox": [327, 1967, 642, 101], "text": "meinden so sehr am Herzen", "project": "MM_1_006"}
{"file": "01531.jpg", "bbox": [330, 2038, 692, 99], "text": "tigung dieses Gegenstandes der", "project": "MM_1_006"}
{"file": "01609.jpg", "bbox": [346, 896, 690, 92], "text": "wieder; aus allen Cantonen rükten", "project": "MM_1_006"}
{"file": "01582.jpg", "bbox": [376, 524, 658, 96], "text": "er Vorschlag des Kirchenraths", "project": "MM_1_006"}
{"file": "01563.jpg", "bbox": [320, 1849, 640, 88], "text": "Cantons Schaffhausen vom 29ten", "project": "MM_1_006"}
{"file": "01485.jpg", "bbox": [265, 1730, 748, 99], "text": "des Cantons Zürich, habe ich vernom¬", "project": "MM_1_006"}
{"file": "01563.jpg", "bbox": [312, 2527, 752, 95], "text": "chend getroffene Verfügung ver¬", "project": "MM_1_006"}
{"file": "01590.jpg", "bbox": [363, 1890, 675, 94], "text": "sich unter der Zahl jener Unglük¬", "project": "MM_1_006"}
{"file": "01649.jpg", "bbox": [308, 2856, 721, 81], "text": "fenen, im Ganzen auf fl. 4246 16 ß", "project": "MM_1_006"}
{"file": "01609.jpg", "bbox": [343, 2008, 675, 95], "text": "blutigen Urtheil nicht entgehen.", "project": "MM_1_006"}
{"file": "01583.jpg", "bbox": [359, 967, 663, 81], "text": "halt und die Beantwortung des", "project": "MM_1_006"}
{"file": "01470.jpg", "bbox": [355, 624, 670, 101], "text": "bedeutende Fehler und falsche Re¬", "project": "MM_1_006"}
{"file": "01502.jpg", "bbox": [331, 2246, 631, 81], "text": "amt geschritten, und hat den,", "project": "MM_1_006"}
{"file": "01556.jpg", "bbox": [356, 2631, 706, 64], "text": "Stadt zu bewilligen, und einem", "project": "MM_1_006"}
{"file": "01526.jpg", "bbox": [352, 649, 695, 70], "text": "nung noch strafbarer? Ist es die", "project": "MM_1_006"}
{"file": "01558.jpg", "bbox": [344, 1722, 736, 98], "text": "In Genehmigung des, von der Fi¬", "project": "MM_1_006"}
{"file": "01474.jpg", "bbox": [347, 605, 752, 97], "text": "ren von Weßemberg [laut Mißi¬", "project": "MM_1_006"}
{"file": "01500.jpg", "bbox": [314, 554, 691, 94], "text": "Gewärtigung der fernern Berichte", "project": "MM_1_006"}
{"file": "01622.jpg", "bbox": [337, 1195, 736, 78], "text": "Summe von 20,000 fl. Reichsvaluta als", "project": "MM_1_006"}
{"file": "01588.jpg", "bbox": [277, 1980, 717, 100], "text": "des Aufruhrs jeden Tag weiter", "project": "MM_1_006"}
{"file": "01576.jpg", "bbox": [361, 2938, 695, 95], "text": "verabfolgt werden möchte, von des", "project": "MM_1_006"}
{"file": "01622.jpg", "bbox": [330, 3002, 749, 94], "text": "rung die weiteren Äußerungen", "project": "MM_1_006"}
{"file": "01650.jpg", "bbox": [329, 2214, 627, 78], "text": "auf die Allment treiben, um", "project": "MM_1_006"}
{"file": "01502.jpg", "bbox": [349, 2547, 648, 87], "text": "Herrn alt Zunftmeister Hs: Ca¬", "project": "MM_1_006"}
{"file": "01604.jpg", "bbox": [339, 957, 732, 113], "text": "Gegenstand und der dießfalls", "project": "MM_1_006"}
{"file": "01530.jpg", "bbox": [335, 1642, 669, 102], "text": "dentlichen Standescommißion zu", "project": "MM_1_006"}
{"file": "01554.jpg", "bbox": [314, 1551, 680, 93], "text": "dem Herrn Oberst Waldner von", "project": "MM_1_006"}
{"file": "01476.jpg", "bbox": [359, 1605, 628, 66], "text": "den Corporationen, und wo", "project": "MM_1_006"}
{"file": "01622.jpg", "bbox": [335, 2703, 728, 96], "text": "ben auf dem Wege der Correspon¬", "project": "MM_1_006"}
{"file": "01460.jpg", "bbox": [337, 2045, 742, 90], "text": "deßelben, die Wahlart der Pfarrer", "project": "MM_1_006"}
{"file": "01653.jpg", "bbox": [339, 2527, 734, 85], "text": "als dem Bibliothek-Convent in Win¬", "project": "MM_1_006"}
{"file": "01660.jpg", "bbox": [323, 2510, 359, 76], "text": "richtiget werden.", "project": "MM_1_006"}
{"file": "01533.jpg", "bbox": [315, 469, 724, 88], "text": "stiz- und Polizeycommißion zu nähe¬", "project": "MM_1_006"}
{"file": "01595.jpg", "bbox": [322, 1635, 738, 92], "text": "dabey thätig bewiesen haben, durch", "project": "MM_1_006"}
{"file": "01621.jpg", "bbox": [363, 637, 703, 105], "text": "den Falschmünzer Johannes Schelt¬", "project": "MM_1_006"}
{"file": "01485.jpg", "bbox": [248, 2298, 765, 101], "text": "ganzen Vaterlandes, der Beyfall Eu¬", "project": "MM_1_006"}
{"file": "01582.jpg", "bbox": [324, 2521, 690, 106], "text": "Heß, und dem Herren Unter¬", "project": "MM_1_006"}
{"file": "01577.jpg", "bbox": [304, 485, 735, 72], "text": "liberg, Erlenbach und den weiter", "project": "MM_1_006"}
{"file": "01506.jpg", "bbox": [355, 900, 676, 103], "text": "und daß die minderschuldigen", "project": "MM_1_006"}
{"file": "01571.jpg", "bbox": [333, 2424, 725, 89], "text": "sammlung zu entlaßen, und die Huldi¬", "project": "MM_1_006"}
{"file": "01649.jpg", "bbox": [342, 418, 723, 92], "text": "dings verbunden sey, das laut Ur¬", "project": "MM_1_006"}
{"file": "01575.jpg", "bbox": [324, 1081, 673, 87], "text": "deßwegen keinen Anstand, die", "project": "MM_1_006"}
{"file": "01612.jpg", "bbox": [344, 2604, 721, 80], "text": "Extraconferenz in Frauenfeld ein¬", "project": "MM_1_006"}
{"file": "01581.jpg", "bbox": [350, 1237, 714, 87], "text": "ben. Geben, Zürich den 20ten Merz 1804.", "project": "MM_1_006"}
{"file": "01610.jpg", "bbox": [355, 1967, 663, 81], "text": "und nochmaliger Verlesung des", "project": "MM_1_006"}
{"file": "01570.jpg", "bbox": [317, 857, 150, 83], "text": "trachte.", "project": "MM_1_006"}
{"file": "01666.jpg", "bbox": [334, 2299, 672, 88], "text": "den verlesen. In dem Schreiben", "project": "MM_1_006"}
{"file": "01507.jpg", "bbox": [316, 858, 694, 110], "text": "unverträglich wäre, eine gütli¬", "project": "MM_1_006"}
{"file": "01531.jpg", "bbox": [329, 744, 695, 104], "text": "dem Ersuchen, in den betreffen¬", "project": "MM_1_006"}
{"file": "01617.jpg", "bbox": [320, 985, 748, 91], "text": "gung ihres mündlich beygefügten", "project": "MM_1_006"}
{"file": "01600.jpg", "bbox": [351, 3027, 730, 111], "text": "Deputation, nach vergeblich erschöpf¬", "project": "MM_1_006"}
{"file": "01524.jpg", "bbox": [334, 1400, 731, 88], "text": "Kosten, dem Kleinen Rath einen aus¬", "project": "MM_1_006"}
{"file": "01571.jpg", "bbox": [318, 2980, 706, 90], "text": "zwar Gemeinds-Abgeordnete den", "project": "MM_1_006"}
{"file": "01611.jpg", "bbox": [301, 3081, 747, 99], "text": "definitiv entschieden seyn wird, zu", "project": "MM_1_006"}
{"file": "01664.jpg", "bbox": [316, 1140, 700, 93], "text": "suspendiert sey, und dieses Geschäft", "project": "MM_1_006"}
{"file": "01460.jpg", "bbox": [342, 750, 749, 92], "text": "einer wohlverdienten Entfernung", "project": "MM_1_006"}
{"file": "01573.jpg", "bbox": [272, 919, 750, 85], "text": "men, daß die von derselben getrof¬", "project": "MM_1_006"}
{"file": "01566.jpg", "bbox": [301, 2756, 763, 96], "text": "man hießigen Orts keine Möglich¬", "project": "MM_1_006"}
{"file": "01537.jpg", "bbox": [311, 3074, 664, 112], "text": "be mit den übrigen Eydsge-", "project": "MM_1_006"}
{"file": "01620.jpg", "bbox": [311, 1913, 704, 88], "text": "halte, daß demselben sowohl die aus¬", "project": "MM_1_006"}
{"file": "01579.jpg", "bbox": [336, 2668, 685, 75], "text": "markt aus einem Ballot, von dem", "project": "MM_1_006"}
{"file": "01633.jpg", "bbox": [312, 1321, 687, 98], "text": "zwischen der Hofgemeinde Illnau", "project": "MM_1_006"}
{"file": "01624.jpg", "bbox": [342, 2877, 732, 111], "text": "Schulbesoldung von Zürich bezogen,", "project": "MM_1_006"}
{"file": "01625.jpg", "bbox": [338, 426, 664, 89], "text": "wiesene Sorgfalt und Thätigkeit", "project": "MM_1_006"}
{"file": "01539.jpg", "bbox": [339, 993, 535, 84], "text": "Seebett hinaus zu rüken.", "project": "MM_1_006"}
{"file": "01650.jpg", "bbox": [327, 1961, 649, 101], "text": "worden, oder solche, die auch in", "project": "MM_1_006"}
{"file": "01636.jpg", "bbox": [329, 1425, 728, 113], "text": "gewesener Daniel Zollinger von", "project": "MM_1_006"}
{"file": "01522.jpg", "bbox": [328, 2573, 709, 97], "text": "te zu dem Huldigungs-Actus in", "project": "MM_1_006"}
{"file": "01491.jpg", "bbox": [321, 957, 749, 87], "text": "mit Mehrheit der Stimmen nachste¬", "project": "MM_1_006"}
{"file": "01565.jpg", "bbox": [328, 1959, 754, 105], "text": "nicht entschließen, auf die gemach¬", "project": "MM_1_006"}
{"file": "01652.jpg", "bbox": [314, 955, 723, 94], "text": "manns vom 7ten dieß von dem Eid¬", "project": "MM_1_006"}
{"file": "01584.jpg", "bbox": [331, 2657, 755, 100], "text": "Großen oder Kleinen Raths gewärtige,", "project": "MM_1_006"}
{"file": "01494.jpg", "bbox": [349, 1670, 731, 73], "text": "den linken Flügel der Rebellen com¬", "project": "MM_1_006"}
{"file": "01547.jpg", "bbox": [331, 1065, 666, 97], "text": "term 29ten pti an die Canzley einge¬", "project": "MM_1_006"}
{"file": "01575.jpg", "bbox": [311, 2282, 676, 93], "text": "der Bürger Stoppani Mitglied", "project": "MM_1_006"}
{"file": "01543.jpg", "bbox": [326, 1532, 649, 81], "text": "ner von Winterthur, worinn", "project": "MM_1_006"}
{"file": "01533.jpg", "bbox": [299, 2681, 730, 73], "text": "biet verübten Criminal-Verbre¬", "project": "MM_1_006"}
{"file": "01514.jpg", "bbox": [277, 2010, 749, 97], "text": "nes von der außerordentlichen Stan¬", "project": "MM_1_006"}
{"file": "01468.jpg", "bbox": [327, 2783, 654, 101], "text": "fälligen Reglements zu hin¬", "project": "MM_1_006"}
{"file": "01477.jpg", "bbox": [279, 884, 745, 93], "text": "nen Beschlußes, heute zu Wieder¬", "project": "MM_1_006"}
{"file": "01550.jpg", "bbox": [317, 2671, 719, 88], "text": "Genügen geleistet hat, – so wurde,", "project": "MM_1_006"}
{"file": "01637.jpg", "bbox": [347, 764, 732, 90], "text": "rem Anhang durch die Polizeywache", "project": "MM_1_006"}
{"file": "01486.jpg", "bbox": [317, 2608, 752, 101], "text": "dem brandbeschädigten Jacob Boß¬", "project": "MM_1_006"}
{"file": "01629.jpg", "bbox": [319, 2742, 726, 85], "text": "ter gestrigem Dato die endlichen", "project": "MM_1_006"}
{"file": "01614.jpg", "bbox": [306, 2934, 710, 81], "text": "erwählten Mitglied deßelben Kennt¬", "project": "MM_1_006"}
{"file": "01532.jpg", "bbox": [316, 626, 741, 82], "text": "storf eine strafbare Betriebsamkeit", "project": "MM_1_006"}
{"file": "01603.jpg", "bbox": [312, 2512, 697, 96], "text": "Pfarrer Zimmermann in Betrach¬", "project": "MM_1_006"}
{"file": "01644.jpg", "bbox": [299, 1018, 748, 89], "text": "Warnung den öffentlichen Blättern", "project": "MM_1_006"}
{"file": "01609.jpg", "bbox": [331, 1954, 701, 85], "text": "Gerechtigkeit, und einem strengen", "project": "MM_1_006"}
{"file": "01550.jpg", "bbox": [314, 1797, 746, 79], "text": "ohne Nachtheil der Sache und des Ver¬", "project": "MM_1_006"}
{"file": "01514.jpg", "bbox": [272, 2188, 748, 102], "text": "Absicht auf die, von dem Herrn Raths¬", "project": "MM_1_006"}
{"file": "01462.jpg", "bbox": [277, 652, 747, 97], "text": "von daselbst, dermalen zu Zeug¬", "project": "MM_1_006"}
{"file": "01499.jpg", "bbox": [329, 1657, 730, 95], "text": "was sie zu Beybehaltung und Wie¬", "project": "MM_1_006"}
{"file": "01618.jpg", "bbox": [329, 2746, 744, 84], "text": "die Huldigung feyerlich, und in", "project": "MM_1_006"}
{"file": "01541.jpg", "bbox": [346, 939, 722, 88], "text": "lich dieser Behörde anhero Bericht zu", "project": "MM_1_006"}
{"file": "01525.jpg", "bbox": [339, 503, 687, 95], "text": "des Argäu anhero communiciert", "project": "MM_1_006"}
{"file": "01653.jpg", "bbox": [343, 687, 722, 75], "text": "4o. Die Regierung des Cantons Ar¬", "project": "MM_1_006"}
{"file": "01633.jpg", "bbox": [315, 909, 290, 57], "text": "veranstalten.", "project": "MM_1_006"}
{"file": "01478.jpg", "bbox": [332, 1931, 687, 91], "text": "5o. In Betref der, zu den Huldi¬", "project": "MM_1_006"}
{"file": "01552.jpg", "bbox": [371, 1534, 665, 88], "text": "von St. Gallen unterm 2ten dieß", "project": "MM_1_006"}
{"file": "01504.jpg", "bbox": [341, 3079, 665, 102], "text": "ter Asper und einigen Mit-", "project": "MM_1_006"}
{"file": "01630.jpg", "bbox": [286, 1153, 742, 105], "text": "Vogel selbst zum persönlichen Schutz", "project": "MM_1_006"}
{"file": "01509.jpg", "bbox": [314, 1361, 737, 89], "text": "Landammann der Schweiz bereit, und", "project": "MM_1_006"}
{"file": "01633.jpg", "bbox": [323, 965, 720, 83], "text": "Über die appellatorio modo an den Klei¬", "project": "MM_1_006"}
{"file": "01658.jpg", "bbox": [340, 813, 720, 105], "text": "dieser Schuldforderungen an den", "project": "MM_1_006"}
{"file": "01578.jpg", "bbox": [333, 740, 385, 87], "text": "hang zu versehen.", "project": "MM_1_006"}
{"file": "01625.jpg", "bbox": [327, 832, 698, 99], "text": "staltern und Führern allfählig die¬", "project": "MM_1_006"}
{"file": "01648.jpg", "bbox": [329, 931, 668, 79], "text": "den Staatswaldungen im Lim¬", "project": "MM_1_006"}
{"file": "01619.jpg", "bbox": [301, 1777, 728, 79], "text": "Rüti bis Uster und von Wald bis Uster", "project": "MM_1_006"}
{"file": "01664.jpg", "bbox": [315, 1083, 683, 77], "text": "St. Gallischer Schulden einstweilen", "project": "MM_1_006"}
{"file": "01626.jpg", "bbox": [324, 1395, 670, 107], "text": "ten Ständen neuerdings aufge¬", "project": "MM_1_006"}
{"file": "01518.jpg", "bbox": [331, 2887, 733, 95], "text": "gen der erforderlichen jährlichen", "project": "MM_1_006"}
{"file": "01636.jpg", "bbox": [328, 3091, 739, 82], "text": "falls ein förmliches Attestat aus-", "project": "MM_1_006"}
{"file": "01627.jpg", "bbox": [327, 2762, 655, 96], "text": "fanden sich bey ihm aus den", "project": "MM_1_006"}
{"file": "01556.jpg", "bbox": [336, 614, 750, 94], "text": "Beylage mit besonderm Intereße,", "project": "MM_1_006"}
{"file": "01551.jpg", "bbox": [339, 2255, 684, 89], "text": "als andere Gemeinden, bei ihren", "project": "MM_1_006"}
{"file": "01562.jpg", "bbox": [294, 2961, 763, 100], "text": "Personals des gedachten Gerichts über¬", "project": "MM_1_006"}
{"file": "01654.jpg", "bbox": [321, 1169, 692, 104], "text": "zu Weinfelden, hat der Kleine Rath,", "project": "MM_1_006"}
{"file": "01640.jpg", "bbox": [271, 2017, 740, 101], "text": "ben aus Versehen nicht beygelegten,", "project": "MM_1_006"}
{"file": "01548.jpg", "bbox": [324, 2695, 686, 108], "text": "Untergeordneten gut sprechen", "project": "MM_1_006"}
{"file": "01561.jpg", "bbox": [322, 3093, 715, 84], "text": "zu machen“, in seinem Schrei-", "project": "MM_1_006"}
{"file": "01624.jpg", "bbox": [330, 952, 740, 88], "text": "sich auszeichnende Lotterie errich¬", "project": "MM_1_006"}
{"file": "01488.jpg", "bbox": [367, 1763, 665, 100], "text": "schein zur Beseitigung dieses Ge¬", "project": "MM_1_006"}
{"file": "01566.jpg", "bbox": [321, 1516, 747, 95], "text": "werden. Eben so wird in Bezug auf", "project": "MM_1_006"}
{"file": "01547.jpg", "bbox": [334, 626, 705, 84], "text": "wird der Commißion des Innern mit", "project": "MM_1_006"}
{"file": "01560.jpg", "bbox": [350, 2656, 656, 98], "text": "digt, ob der dortige Bürger und", "project": "MM_1_006"}
{"file": "01585.jpg", "bbox": [328, 541, 685, 94], "text": "standen darin, daß zween Reprä¬", "project": "MM_1_006"}
{"file": "01578.jpg", "bbox": [352, 3011, 724, 85], "text": "Ruhestöhrer aber, gegen ein stür¬", "project": "MM_1_006"}'''
if _EVAL500_HITS:
    EVAL500_ROWS = [json.loads(l) for l in Path(_EVAL500_HITS[0]).open()]
else:
    print("STAGE eval500: using embedded blob", flush=True)
    EVAL500_ROWS = [json.loads(l) for l in _EVAL500_BLOB.strip().splitlines()]
assert len(EVAL500_ROWS) == 500, f"eval500 rows {len(EVAL500_ROWS)}"
assert all(r.get("project") == "MM_1_006" for r in EVAL500_ROWS), "eval500 blob project mismatch"
OUT = Path(os.environ.get("KURRENT_OUT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "dh-unibe/trocr-kurrent"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if SMOKE:
    EPOCHS, BATCH, EVAL_EVERY, WARMUP, LR = 1, 1, int(os.environ.get("KURRENT_SMOKE_EVAL_EVERY", "1")), 0, 1e-5
    MAX_STEPS = int(os.environ.get("KURRENT_SMOKE_STEPS", "1"))
else:
    EPOCHS, BATCH, EVAL_EVERY, WARMUP, LR = 2, 8, 125, 100, 5e-6
    MAX_STEPS = 500  # v16: v15 proved best@500 then overfit (names 0.889->1.080); cap at the productive regime
PAD = 6
print(f"STAGE imports-done device={DEVICE} gpu={torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'none'} smoke={SMOKE}", flush=True)

# --- run-progress heartbeat (issue #30): best-effort, never raises ---
PROGRESS_DATASET = os.environ.get("KURRENT_PROGRESS_DATASET", "reihlinstinct/kurrent-run-progress")
PROGRESS_ENABLED = os.environ.get("KURRENT_PROGRESS", "1") == "1"
_progress_state = {"pushes": 0, "last_error": None}


def progress_payload(stage: str, **fields) -> dict:
    payload = {"stage": stage, "elapsed_sec": round(time.time() - T0, 1)}
    payload.update(fields)
    return payload


def _progress_api():
    """Authenticated Kaggle API: in-kernel session first, then KAGGLE_API_TOKEN
    (e.g. injected as a Kaggle User Secret)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _json_safe(value):
    """Replace non-finite floats with None so json.dumps never emits Infinity/NaN."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def progress_push(payload: dict) -> bool:
    """Version the PRIVATE progress dataset with this payload as progress.json.

    Best-effort: returns False instead of raising so training never dies on a
    heartbeat failure. Payload policy (issue #30 review): aggregates only, no
    per-row hypotheses or gold text.
    """
    if not PROGRESS_ENABLED:
        return False
    import shutil
    import tempfile
    d = None
    try:
        d = Path(tempfile.mkdtemp(prefix="kurrent-progress-"))
        payload = _json_safe(payload)
        (d / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "dataset-metadata.json").write_text(json.dumps({
            "title": PROGRESS_DATASET.split("/")[-1],
            "id": PROGRESS_DATASET,
            "licenses": [{"name": "other"}],
        }), encoding="utf-8")
        _progress_api().dataset_create_version(str(d), str(payload.get("stage", "progress")), dir_mode="skip")
        _progress_state["pushes"] += 1
        print(f"PROGRESS pushed: {payload.get('stage')} (total {_progress_state['pushes']})", flush=True)
        return True
    except Exception as e:
        _progress_state["last_error"] = type(e).__name__
        print(f"PROGRESS push failed (non-fatal): {_progress_state['last_error']}", flush=True)
        return False
    finally:
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)



print("STAGE processor-load", flush=True)
processor = TrOCRProcessor.from_pretrained(MODEL_ID)
CKPT_SRC = os.environ.get("KURRENT_CKPT_SRC", "/kaggle/input/__nonexistent_no_checkpoint__")  # from-base by design (v8)
print("STAGE model-download", flush=True)
if os.path.exists(os.path.join(CKPT_SRC, "config.json")):
    model = VisionEncoderDecoderModel.from_pretrained(CKPT_SRC).to(DEVICE)
    print(f"STAGE model-ready src={CKPT_SRC} (resume from v2 best)", flush=True)
else:
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE)
    print(f"STAGE model-ready src={MODEL_ID} (BASE - checkpoint missing at {CKPT_SRC}!)", flush=True)
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# --- v13 gentle-training block (issue #37): exactly one variant per run ---
GENTLE = os.environ.get("KURRENT_GENTLE", "lowlr")  # v20: arm-B data, lowlr (v18 recipe; v19 freeze-encoder regressed)
if GENTLE not in ("lowlr", "freeze-encoder"):
    raise ValueError(f"unknown KURRENT_GENTLE variant: {GENTLE!r}")
if not SMOKE:
    if GENTLE == "lowlr":
        LR = 1e-6  # was 5e-6 in v8-v12
    else:  # freeze-encoder
        for _p in model.encoder.parameters():
            _p.requires_grad_(False)
_frozen = sum(1 for _p in model.parameters() if not _p.requires_grad)
print(f"STAGE gentle mode={GENTLE} lr={LR} frozen_params={_frozen}", flush=True)


def cer(hyp: str, ref: str) -> float:
    rows, cols = len(hyp) + 1, len(ref) + 1
    d = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        d[i][0] = i
    for j in range(cols):
        d[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]))
    return d[-1][-1] / max(1, len(ref))


def word_acc(hyp: str, ref: str) -> float:
    hw, rw = hyp.split(), ref.split()
    return sum(1 for w in rw if w in hw) / max(1, len(rw))


def read_pil(img: Image.Image) -> str:
    pixels = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(pixels, max_length=64)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


_page_cache: dict = {}

def crop_from_page(row: dict) -> Image.Image:
    img = _page_cache.get(row["file"])
    if img is None:
        img = Image.open(PAGE_ROOT[row["file"]]).convert("RGB")
        if len(_page_cache) >= 4:
            _page_cache.pop(next(iter(_page_cache)))
        _page_cache[row["file"]] = img
    x, y, w, h = row["bbox"]
    x0 = max(0, x - PAD)
    y0 = max(0, y - PAD)
    x1 = min(img.width, x + w + PAD)
    y1 = min(img.height, y + h + PAD)
    return img.crop((x0, y0, x1, y1))


def _safe_data_path(root, rel):
    """Resolve rel under root, rejecting paths that escape it."""
    base = root.resolve()
    p = (base / rel).resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"unsafe path escapes data root: {rel!r}")
    return p


def evaluate12(tag: str) -> dict:
    rows = [json.loads(l) for l in (DATA / "eval_manifest.jsonl").open()]
    results = {}
    model.eval()
    for r in rows:
        hyp = read_pil(Image.open(_safe_data_path(DATA, r["file"])))
        entry = {"hyp": hyp}
        gold = r.get("scored_gold") or ""
        if r["id"] in ("h01", "h07") and gold:
            entry["cer"] = round(cer(hyp, gold), 3)
            entry["word_acc"] = round(word_acc(hyp, gold), 3)
        elif gold:
            if isinstance(gold, list):
                entry["span_hit"] = {g: (g in hyp) for g in gold}
            else:
                entry["span_hit"] = gold in hyp
        results[r["id"]] = entry
    print(f"EVAL12 {tag}: {json.dumps(results, ensure_ascii=False)}", flush=True)
    return results


def score12(results: dict) -> float:
    vals = [results[k]["cer"] for k in ("h01", "h07") if "cer" in results.get(k, {})]
    return sum(vals) / len(vals) if vals else float("inf")


_EVAL_NAMES_BLOB = '''{"file": "s05_22.jpg", "bbox": [630, 1790, 260, 130], "text": "Schächter", "project": "eschau-occ", "kind": "occupation", "IdNr": 38666, "eval_src": "v4-original"}
{"file": "s07_22.jpg", "bbox": [462, 1066, 166, 137], "text": "Metzger", "project": "eschau-occ", "kind": "occupation", "IdNr": 38678, "eval_src": "v4-original"}
{"file": "s07_22.jpg", "bbox": [462, 1560, 166, 108], "text": "Makler", "project": "eschau-occ", "kind": "occupation", "IdNr": 38693, "eval_src": "v4-original"}
{"file": "s07_22.jpg", "bbox": [462, 1870, 166, 183], "text": "Lehrer", "project": "eschau-occ", "kind": "occupation", "IdNr": 38696, "eval_src": "v4-original"}
{"file": "s04_22.jpg", "bbox": [150, 995, 280, 170], "text": "Gerson Mosbacher", "project": "eschau-gerson", "kind": "name", "IdNr": 33739, "eval_src": "v4-original"}
{"file": "s04_22.jpg", "bbox": [890, 995, 360, 170], "text": "28.05.1845", "scored_gold": ["28", "1845"], "project": "eschau-gerson", "kind": "date", "IdNr": 33739, "eval_src": "v4-original"}
{"file": "s04_22.jpg", "bbox": [537, 995, 241, 170], "text": "Handelsmann", "project": "eschau-gerson", "kind": "occupation", "IdNr": 33739, "eval_src": "v4-original"}
{"file": "s05_22.jpg", "bbox": [223, 1795, 231, 159], "text": "Michel Simson Mayer", "project": "eschau-seeds", "kind": "name", "IdNr": 38666, "conf": "index-text", "eval_src": "existing-heldout-person"}
{"file": "s05_22.jpg", "bbox": [652, 1821, 193, 151], "text": "20.01.1855", "project": "eschau-seeds", "kind": "date", "IdNr": 38666, "conf": "index-text", "eval_src": "existing-heldout-person", "scored_gold": ["20", "1855"]}
{"file": "s07_22.jpg", "bbox": [150, 1066, 308, 137], "text": "Rothschild Hirsch", "project": "eschau-seeds", "kind": "name", "IdNr": 38678, "conf": "index-text", "eval_src": "existing-heldout-person"}
{"file": "s07_22.jpg", "bbox": [1400, 1092, 330, 129], "text": "10.10.1864", "project": "eschau-seeds", "kind": "date", "IdNr": 38678, "conf": "index-text", "eval_src": "existing-heldout-person", "scored_gold": ["10", "1864"]}
{"file": "s07_22.jpg", "bbox": [150, 1560, 308, 108], "text": "Rothschild Moses", "project": "eschau-seeds", "kind": "name", "IdNr": 38693, "conf": "index-text", "eval_src": "existing-heldout-person"}
{"file": "s07_22.jpg", "bbox": [1400, 1586, 330, 100], "text": "30.04.1867", "project": "eschau-seeds", "kind": "date", "IdNr": 38693, "conf": "index-text", "eval_src": "existing-heldout-person", "scored_gold": ["30", "1867"]}
{"file": "s07_22.jpg", "bbox": [150, 1870, 308, 183], "text": "Marcus Ascher", "project": "eschau-seeds", "kind": "name", "IdNr": 38696, "conf": "index-text", "eval_src": "existing-heldout-person"}
{"file": "s07_22.jpg", "bbox": [1400, 1896, 330, 175], "text": "14.04.1869", "project": "eschau-seeds", "kind": "date", "IdNr": 38696, "conf": "index-text", "eval_src": "existing-heldout-person", "scored_gold": ["14", "1869"]}
{"file": "g04_22.jpg", "bbox": [150, 540, 280, 142], "text": "Mendel Mosbacher", "IdNr": 3763, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n3763", "note": "duplicate index person 3764 = same entry, labeled once (3763); the g12/g13 father as a CHILD here (b.18.10.1831)", "eval_src": "new-heldout-person"}
{"file": "g04_22.jpg", "bbox": [2100, 541, 400, 139], "text": "18.10.1831", "IdNr": 3763, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d3763", "note": "'d 18t Oct: 1831 Morgens 8 Uhr' legible", "eval_src": "new-heldout-person", "scored_gold": ["18", "1831"]}
{"file": "g11_22.jpg", "bbox": [235, 720, 200, 95], "text": "Felix Mosbacher", "IdNr": 3766, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r2", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person"}
{"file": "g11_22.jpg", "bbox": [1740, 735, 420, 150], "text": "18.03.1864", "IdNr": 3766, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1864", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person", "scored_gold": ["18", "1864"]}
{"file": "g04_22.jpg", "bbox": [150, 1450, 280, 115], "text": "Joseph Mosbacher", "IdNr": 6403, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6403", "note": "duplicate index person 38095 = same entry, labeled once (6403); later-hand death note '+ d. 9 Aug. 1835' below name EXCLUDED from bbox", "eval_src": "new-heldout-person"}
{"file": "g04_22.jpg", "bbox": [2100, 1252, 400, 152], "text": "18.08.1833", "IdNr": 6403, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6403", "note": "'d 18t Aug 1833 Abends 6(?) Uhr' legible", "eval_src": "new-heldout-person", "scored_gold": ["18", "1833"]}
{"file": "g12_22.jpg", "bbox": [70, 1850, 390, 135], "text": "Lina Mosbacher", "IdNr": 6412, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6412", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person"}
{"file": "g12_22.jpg", "bbox": [1560, 1614, 900, 115], "text": "15.08.1872", "IdNr": 6412, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6412", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Donnerstag OK", "eval_src": "new-heldout-person", "scored_gold": ["15", "1872"]}
{"file": "g11_22.jpg", "bbox": [210, 1200, 230, 135], "text": "Louis Mosbacher", "IdNr": 6446, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r4", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person"}
{"file": "g11_22.jpg", "bbox": [1740, 1210, 420, 165], "text": "14.09.1866", "IdNr": 6446, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1866", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person", "scored_gold": ["14", "1866"]}
{"file": "g12_22.jpg", "bbox": [70, 905, 390, 95], "text": "Rickchen Mosbacher", "IdNr": 6449, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6449", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person"}
{"file": "g12_22.jpg", "bbox": [1560, 950, 900, 120], "text": "02.06.1870", "IdNr": 6449, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6449", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Donnerstag OK", "eval_src": "new-heldout-person", "scored_gold": ["2", "1870"]}
{"file": "g12_22.jpg", "bbox": [70, 1615, 390, 85], "text": "Zerline Mosbacher", "IdNr": 6500, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6500", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person"}
{"file": "g12_22.jpg", "bbox": [1560, 1865, 900, 130], "text": "18.06.1872", "IdNr": 6500, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6500", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; cell reads 'Dienstag den 8ten[struck] achtzehnten Juni' + 1897 margin correction confirming 18; Dienstag OK for 18.06.1872", "eval_src": "new-heldout-person", "scored_gold": ["18", "1872"]}
{"file": "s06_22.jpg", "bbox": [140, 794, 420, 220], "text": "Alexander Strauß", "IdNr": 38035, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.3", "eval_src": "new-heldout-person"}
{"file": "s06_22.jpg", "bbox": [1290, 900, 240, 124], "text": "19.10.1857", "IdNr": 38035, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.5", "eval_src": "new-heldout-person", "scored_gold": ["19", "1857"]}
{"file": "g04_22.jpg", "bbox": [150, 365, 280, 135], "text": "Veila Mosbacher", "IdNr": 38037, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38037", "note": "ink reads 'Veile', index 'Veila'; label from index", "eval_src": "new-heldout-person"}
{"file": "g04_22.jpg", "bbox": [2100, 370, 400, 146], "text": "23.04.1831", "IdNr": 38037, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38037", "note": "'d 23t Apr: 1831 Morgens 1 Uhr' legible; no weekday words in this register", "eval_src": "new-heldout-person", "scored_gold": ["23", "1831"]}
{"file": "g04_22.jpg", "bbox": [150, 1880, 280, 137], "text": "Madel Mosbacher", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38098", "note": "d.08.04.1837 per index (also s03 death register person)", "eval_src": "new-heldout-person"}
{"file": "g04_22.jpg", "bbox": [2100, 1924, 400, 176], "text": "28.11.1835", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38098", "note": "'d 28t Nov 1835 Abends 9 Uhr' legible", "eval_src": "new-heldout-person", "scored_gold": ["28", "1835"]}
{"file": "s03_22.jpg", "bbox": [147, 1838, 226, 164], "text": "Madel Mosbacher", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.11", "eval_src": "new-heldout-person"}
{"file": "s03_22.jpg", "bbox": [1292, 1837, 442, 167], "text": "08.04.1837", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.11", "eval_src": "new-heldout-person", "scored_gold": ["8", "1837"]}
{"file": "g05_22.jpg", "bbox": [160, 171, 240, 382], "text": "Rebecca Mosbacher", "IdNr": 38099, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.0-1", "eval_src": "new-heldout-person"}
{"file": "g05_22.jpg", "bbox": [2050, 170, 420, 200], "text": "10.01.1836", "IdNr": 38099, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.1-2", "eval_src": "new-heldout-person", "scored_gold": ["10", "1836"]}
{"file": "g05_22.jpg", "bbox": [160, 1317, 240, 191], "text": "Aaron Strauß", "IdNr": 38106, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6", "eval_src": "new-heldout-person"}
{"file": "g05_22.jpg", "bbox": [2050, 1288, 420, 179], "text": "14.09.1841", "IdNr": 38106, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.7", "eval_src": "new-heldout-person", "scored_gold": ["14", "1841"]}
{"file": "g05_22.jpg", "bbox": [160, 1699, 240, 191], "text": "Emanuel Mosbacher", "IdNr": 38113, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.8", "eval_src": "new-heldout-person"}
{"file": "g05_22.jpg", "bbox": [2050, 1646, 420, 179], "text": "28.01.1843", "IdNr": 38113, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.9", "eval_src": "new-heldout-person", "scored_gold": ["28", "1843"]}
{"file": "s02_22.jpg", "bbox": [157, 495, 235, 164], "text": "Giedle Mosbacher", "IdNr": 38315, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R3@sameY", "eval_src": "new-heldout-person"}
{"file": "s02_22.jpg", "bbox": [1296, 495, 474, 164], "text": "28.06.1823", "IdNr": 38315, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R3", "eval_src": "new-heldout-person", "scored_gold": ["28", "1823"]}
{"file": "s02_22.jpg", "bbox": [157, 659, 235, 164], "text": "Salomon Mosbacher", "IdNr": 38622, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R4@sameY", "eval_src": "new-heldout-person"}
{"file": "s02_22.jpg", "bbox": [1296, 659, 474, 164], "text": "04.10.1823", "IdNr": 38622, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R4", "eval_src": "new-heldout-person", "scored_gold": ["4", "1823"]}
{"file": "s02_22.jpg", "bbox": [157, 823, 235, 164], "text": "Riefke Rothschild", "IdNr": 38623, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R5@sameY", "eval_src": "new-heldout-person"}
{"file": "s02_22.jpg", "bbox": [1296, 823, 474, 164], "text": "09.04.1825", "IdNr": 38623, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R5", "eval_src": "new-heldout-person", "scored_gold": ["9", "1825"]}
{"file": "s02_22.jpg", "bbox": [157, 987, 235, 164], "text": "Löw Rothschild", "IdNr": 38624, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R6@sameY", "eval_src": "new-heldout-person"}
{"file": "s02_22.jpg", "bbox": [1296, 987, 474, 164], "text": "11.05.1825", "IdNr": 38624, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R6", "eval_src": "new-heldout-person", "scored_gold": ["11", "1825"]}
{"file": "s02_22.jpg", "bbox": [157, 1807, 235, 164], "text": "Lorenz Mosbacher", "IdNr": 38627, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R11@sameY", "eval_src": "new-heldout-person"}
{"file": "s02_22.jpg", "bbox": [1296, 1807, 474, 164], "text": "29.03.1829", "IdNr": 38627, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R11", "eval_src": "new-heldout-person", "scored_gold": ["29", "1829"]}
{"file": "s02_22.jpg", "bbox": [157, 1971, 235, 164], "text": "Michael Michel", "IdNr": 38628, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R12@sameY", "eval_src": "new-heldout-person"}
{"file": "s02_22.jpg", "bbox": [1296, 1971, 474, 164], "text": "21.12.1829", "IdNr": 38628, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R12", "eval_src": "new-heldout-person", "scored_gold": ["21", "1829"]}
{"file": "s03_22.jpg", "bbox": [147, 1018, 226, 164], "text": "Jakob Mehnl", "IdNr": 38636, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6", "eval_src": "new-heldout-person"}
{"file": "s03_22.jpg", "bbox": [1292, 1002, 442, 167], "text": "22.05.1834", "IdNr": 38636, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.6", "eval_src": "new-heldout-person", "scored_gold": ["22", "1834"]}
{"file": "s03_22.jpg", "bbox": [147, 1510, 226, 164], "text": "Jonas Mosbacher", "IdNr": 38639, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.9", "eval_src": "new-heldout-person"}
{"file": "s03_22.jpg", "bbox": [1292, 1503, 442, 167], "text": "09.08.1835", "IdNr": 38639, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.9", "eval_src": "new-heldout-person", "scored_gold": ["9", "1835"]}
{"file": "s03_22.jpg", "bbox": [147, 1674, 226, 164], "text": "Michelle Michael", "IdNr": 38640, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.10", "eval_src": "new-heldout-person"}
{"file": "s03_22.jpg", "bbox": [1292, 1670, 442, 167], "text": "26.06.1836", "IdNr": 38640, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.10", "eval_src": "new-heldout-person", "scored_gold": ["26", "1836"]}
{"file": "s06_22.jpg", "bbox": [140, 1454, 420, 220], "text": "Seligmann Hirsch", "IdNr": 38672, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6", "eval_src": "new-heldout-person"}
{"file": "s06_22.jpg", "bbox": [1290, 1322, 240, 149], "text": "11.08.1858", "IdNr": 38672, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.8", "eval_src": "new-heldout-person", "scored_gold": ["11", "1858"]}
{"file": "g08_22.jpg", "bbox": [150, 1205, 355, 90], "text": "Elias Michel", "IdNr": 38740, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38740", "eval_src": "new-heldout-person"}
{"file": "g08_22.jpg", "bbox": [2100, 1208, 560, 195], "text": "am Donnerstag d. 28ten September 1854, Morgens 4 Uhr", "IdNr": 38740, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38740", "note": "weekday cross-check PASS (28.09.1854 = Thursday)", "eval_src": "new-heldout-person", "scored_gold": ["28te"]}
{"file": "g07_22.jpg", "bbox": [81, 690, 271, 190], "text": "Karoline Mosbacher", "IdNr": 38744, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.3-top", "eval_src": "new-heldout-person"}
{"file": "g07_22.jpg", "bbox": [1908, 690, 348, 190], "text": "12.07.1849", "IdNr": 38744, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "0.3-top", "eval_src": "new-heldout-person", "scored_gold": ["12", "1849"]}
{"file": "g08_22.jpg", "bbox": [150, 530, 355, 90], "text": "Gittel Mosbacher", "IdNr": 38745, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38745", "note": "index name 'Mosbacher, Gitta,Getta (Gittel Gattin des Menachem)'; label matches ink 'Gittel'", "eval_src": "new-heldout-person"}
{"file": "g08_22.jpg", "bbox": [2100, 542, 560, 130], "text": "Montag den 21. Juni 1852, Mittags 4 Uhr", "IdNr": 38745, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38745", "note": "weekday cross-check PASS (21.06.1852 = Monday)", "eval_src": "new-heldout-person", "scored_gold": ["Juni"]}
{"file": "g09_22.jpg", "bbox": [210, 1595, 280, 100], "text": "Maier Joseph Mosbacher", "IdNr": 38746, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38746", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; two-line 'Maier Joseph / Mosbacher'", "eval_src": "new-heldout-person"}
{"file": "g09_22.jpg", "bbox": [1900, 1610, 700, 190], "text": "09.03.1857", "IdNr": 38746, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38746", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; 'Am Montag ... Maerz 1857, Nachts 12 Uhr' - Montag legible, 09.03.1857 = Monday PASS; day numeral from index, pinned by weekday+month+year", "eval_src": "new-heldout-person", "scored_gold": ["9", "1857"]}
{"file": "g08_22.jpg", "bbox": [150, 985, 355, 130], "text": "Gidel Mosbacher", "IdNr": 38751, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38751", "note": "first line struck through by scribe (false start reading '...Strauss...'); label 'Gidel Mosbacher'", "eval_src": "new-heldout-person"}
{"file": "g08_22.jpg", "bbox": [2100, 1042, 560, 150], "text": "am Freitag d. 18ten August 1854, Morgens 4 Uhr", "IdNr": 38751, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38751", "note": "weekday cross-check PASS (18.08.1854 = Friday)", "eval_src": "new-heldout-person", "scored_gold": ["18te"]}
{"file": "g13_22.jpg", "bbox": [230, 815, 350, 115], "text": "Jeta Mosbacher", "IdNr": 38754, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38754", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check", "eval_src": "new-heldout-person"}
{"file": "g13_22.jpg", "bbox": [1520, 840, 940, 140], "text": "21.12.1873", "IdNr": 38754, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38754", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Sonntag OK", "eval_src": "new-heldout-person", "scored_gold": ["21", "1873"]}
{"file": "g13_22.jpg", "bbox": [230, 1520, 350, 125], "text": "Felix Mosbacher", "IdNr": 38755, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38755", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check", "eval_src": "new-heldout-person"}
{"file": "g13_22.jpg", "bbox": [1520, 1575, 940, 110], "text": "25.08.1875", "IdNr": 38755, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38755", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Mittwoch OK", "eval_src": "new-heldout-person", "scored_gold": ["25", "1875"]}
{"file": "g07_22.jpg", "bbox": [81, 1606, 271, 138], "text": "Sarah Rothschild", "IdNr": 38780, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.7", "eval_src": "new-heldout-person"}
{"file": "g07_22.jpg", "bbox": [1908, 1594, 348, 137], "text": "08.04.1851", "IdNr": 38780, "project": "eschau1081-batch1", "kind": "date", "conf": "verified", "band": "1.8", "eval_src": "new-heldout-person", "scored_gold": ["8", "1851"]}
{"file": "g13_22.jpg", "bbox": [230, 1365, 350, 125], "text": "Selma Strauß", "IdNr": 38782, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38782", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check", "eval_src": "new-heldout-person"}
{"file": "g13_22.jpg", "bbox": [1520, 1455, 940, 110], "text": "07.05.1875", "IdNr": 38782, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38782", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Freitag OK", "eval_src": "new-heldout-person", "scored_gold": ["7", "1875"]}
{"file": "g13_22.jpg", "bbox": [230, 630, 350, 140], "text": "Jeanette Strauß", "IdNr": 38783, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38783", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check", "eval_src": "new-heldout-person"}
{"file": "g13_22.jpg", "bbox": [1520, 700, 940, 140], "text": "21.10.1873", "IdNr": 38783, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38783", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Dienstag OK", "eval_src": "new-heldout-person", "scored_gold": ["21", "1873"]}
{"file": "g09_22.jpg", "bbox": [250, 1250, 130, 70], "text": "David", "IdNr": 38786, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38786", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; given name only as inked; surname Strauss in shared parents brace (cert N.104), twins with 38787", "eval_src": "new-heldout-person"}
{"file": "g09_22.jpg", "bbox": [1900, 1215, 700, 240], "text": "20.10.1856", "IdNr": 38786, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38786", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; SHARED date cell with twin 38787; 'Am Montag den zwanzigsten October 1856' - Montag legible, 20.10.1856 = Monday PASS", "eval_src": "new-heldout-person", "scored_gold": ["20", "1856"]}
{"file": "g09_22.jpg", "bbox": [245, 1340, 175, 70], "text": "Jeanette", "IdNr": 38787, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38787", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; given name only as inked; surname Strauss in shared parents brace, twins with 38786", "eval_src": "new-heldout-person"}
{"file": "g09_22.jpg", "bbox": [1900, 1215, 700, 240], "text": "20.10.1856", "IdNr": 38787, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38787", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; SHARED date cell with twin 38786; 'Am Montag den zwanzigsten October 1856' - Montag legible, 20.10.1856 = Monday PASS", "eval_src": "new-heldout-person", "scored_gold": ["20", "1856"]}
{"file": "g11_22.jpg", "bbox": [225, 460, 220, 110], "text": "Sara Strauß", "IdNr": 38789, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r1", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person"}
{"file": "g11_22.jpg", "bbox": [1740, 490, 420, 210], "text": "15.09.1863", "IdNr": 38789, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1863", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person", "scored_gold": ["15", "1863"]}
{"file": "g13_22.jpg", "bbox": [230, 1020, 350, 120], "text": "Moses Strauß", "IdNr": 38792, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38792", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check", "eval_src": "new-heldout-person"}
{"file": "g13_22.jpg", "bbox": [1520, 1100, 940, 130], "text": "06.01.1874", "IdNr": 38792, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38792", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Dienstag OK", "eval_src": "new-heldout-person", "scored_gold": ["6", "1874"]}
{"file": "g13_22.jpg", "bbox": [230, 1670, 350, 115], "text": "Friederike Strauß", "IdNr": 38793, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38793", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check", "eval_src": "new-heldout-person"}
{"file": "g13_22.jpg", "bbox": [1520, 1695, 940, 110], "text": "19.09.1875", "IdNr": 38793, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38793", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Sonntag OK", "eval_src": "new-heldout-person", "scored_gold": ["19", "1875"]}
{"file": "g06_22.jpg", "bbox": [160, 1134, 240, 183], "text": "Edel Strauß", "IdNr": 38796, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6", "eval_src": "new-heldout-person"}
{"file": "g06_22.jpg", "bbox": [2050, 1180, 420, 190], "text": "11.11.1846", "IdNr": 38796, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.6", "eval_src": "new-heldout-person", "scored_gold": ["11", "1846"]}
{"file": "g09_22.jpg", "bbox": [210, 1838, 180, 52], "text": "Babette", "IdNr": 38800, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38800", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; given name only as inked ('posthuma' entry); surname Strauss from index/parent phrase", "eval_src": "new-heldout-person"}
{"file": "g09_22.jpg", "bbox": [1900, 1825, 750, 125], "text": "08.11.1857", "IdNr": 38800, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38800", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; 'Am Sonntag d. 8ten Nov. 1857 Abends' - Sonntag legible, 08.11.1857 = Sunday PASS", "eval_src": "new-heldout-person", "scored_gold": ["8", "1857"]}
{"file": "g11_22.jpg", "bbox": [235, 890, 300, 140], "text": "Jeanette Strauß", "IdNr": 38806, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r3", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person"}
{"file": "g11_22.jpg", "bbox": [1740, 955, 420, 240], "text": "30.08.1865", "IdNr": 38806, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1865", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person", "scored_gold": ["30", "1865"]}
{"file": "g11_22.jpg", "bbox": [185, 1475, 220, 135], "text": "David Strauß", "IdNr": 38807, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r5", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person"}
{"file": "g11_22.jpg", "bbox": [1740, 1505, 420, 150], "text": "06.07.1867", "IdNr": 38807, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1867", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person", "scored_gold": ["6", "1867"]}
{"file": "g12_22.jpg", "bbox": [70, 1735, 390, 90], "text": "Regine Strauß", "IdNr": 38809, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38809", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag", "eval_src": "new-heldout-person"}
{"file": "g12_22.jpg", "bbox": [1560, 1743, 900, 102], "text": "05.10.1872", "IdNr": 38809, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38809", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Samstag OK", "eval_src": "new-heldout-person", "scored_gold": ["5", "1872"]}
{"file": "g13_22.jpg", "bbox": [230, 1195, 350, 110], "text": "Ruben Strauß", "IdNr": 38810, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38810", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check", "eval_src": "new-heldout-person"}
{"file": "g13_22.jpg", "bbox": [1520, 1250, 940, 100], "text": "26.02.1874", "IdNr": 38810, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38810", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Donnerstag OK", "eval_src": "new-heldout-person", "scored_gold": ["26", "1874"]}'''


V4_SUBSET_KEYS = {"occupation:38666", "occupation:38678", "occupation:38693", "occupation:38696", "name:33739", "date:33739", "occupation:33739"}


def evaluate_names(tag: str) -> dict:
    path = CODE / "eval_names_v5.jsonl"
    if path.exists():
        rows = [json.loads(l) for l in path.open()]
    else:
        print(f"EVAL_NAMES {tag}: using embedded blob", flush=True)
        rows = [json.loads(l) for l in _EVAL_NAMES_BLOB.strip().splitlines()]
    assert len(rows) == 105, f"names eval should be 105 rows, got {len(rows)}"
    if SMOKE:
        rows = rows[:1]
    model.eval()
    cers, waccs, detail = [], [], {}
    for r in rows:
        hyp = read_pil(crop_from_page(r))
        key = f'{r.get("kind","?")}:{r.get("IdNr","?")}'
        entry = {"hyp": hyp}
        gold_spans = r.get("scored_gold")
        if gold_spans:
            entry["span_hit"] = {g: (g in hyp) for g in gold_spans}
        else:
            c, w = cer(hyp, r["text"]), word_acc(hyp, r["text"])
            entry["cer"] = round(c, 3)
            entry["word_acc"] = round(w, 3)
            cers.append(c)
            waccs.append(w)
        detail[key] = entry
    out = {"n": len(rows), "cer": round(sum(cers) / max(1, len(cers)), 4), "word_acc": round(sum(waccs) / max(1, len(waccs)), 4), "detail": detail}
    sub = [v["cer"] for k, v in detail.items() if k in V4_SUBSET_KEYS and "cer" in v]
    out["cer_v4subset"] = round(sum(sub) / len(sub), 4) if sub else None
    spans = [h for v in detail.values() for h in (v.get("span_hit") or {}).values()]
    out["span_hit_rate"] = round(sum(1 for h in spans if h) / len(spans), 4) if spans else None
    print(f"EVAL_NAMES {tag}: {json.dumps(out, ensure_ascii=False)}", flush=True)
    return out


def evaluate500(tag: str) -> dict:
    rows = EVAL500_ROWS
    if SMOKE:
        rows = rows[:1]
    model.eval()
    cers, waccs = [], []
    for r in rows:
        hyp = read_pil(crop_from_page(r))
        cers.append(cer(hyp, r["text"]))
        waccs.append(word_acc(hyp, r["text"]))
    out = {"n": len(rows), "cer": round(sum(cers) / max(1, len(cers)), 4), "word_acc": round(sum(waccs) / max(1, len(waccs)), 4)}
    print(f"EVAL500 {tag}: {json.dumps(out)}", flush=True)
    return out


assert (CODE / "train_manifest_v4.jsonl").exists(), f"train_manifest_v4.jsonl missing in {CODE}"
print("STAGE eval-before", flush=True)
base12 = evaluate12("before")
base500 = evaluate500("before")
basenames = evaluate_names("before")

import random as _rnd

_held_idnrs = {"33739", "35238", "3763", "3766", "38035", "38037", "38098", "38099", "38106", "38113", "38315", "38622", "38623", "38624", "38627", "38628", "38636", "38639", "38640", "38666", "38672", "38678", "38693", "38696", "38740", "38744", "38745", "38746", "38751", "38754", "38755", "38780", "38782", "38783", "38786", "38787", "38789", "38792", "38793", "38796", "38800", "38806", "38807", "38809", "38810", "6403", "6412", "6446", "6449", "6500"}  # v5: Gerson row + 49 EVAL_NAMES persons (user-approved 2026-11-08)
# v15: single clean merged manifest (repo labeling/batch2/merged_v15_manifest.json)
# replaces the embedded blob + batch-1 mount + batch-2 dataset v1 (see header).
_merged_path = "EMBEDDED arm-B manifest (repo labeling/batch2/merged_armb_manifest.json)"
_ARMB_BLOB = '''{"file": "g01_22.jpg", "bbox": [310, 512, 300, 48], "text": "Beile Marx", "IdNr": 38736, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1821#1"}
{"file": "g01_22.jpg", "bbox": [1055, 815, 155, 60], "text": "Joseph Mosbacher", "IdNr": 33729, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1821#2"}
{"file": "g01_22.jpg", "bbox": [460, 640, 330, 52], "text": "23.11.1821", "IdNr": 33729, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1821#2"}
{"file": "g01_22.jpg", "bbox": [560, 815, 185, 60], "text": "Merjem Mosbacher", "IdNr": 38747, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1821#2"}
{"file": "g01_22.jpg", "bbox": [460, 640, 330, 52], "text": "23.11.1821", "IdNr": 38747, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1821#2"}
{"file": "g01_22.jpg", "bbox": [940, 1105, 170, 43], "text": "Beile Michel", "IdNr": 38737, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1822#1"}
{"file": "g01_22.jpg", "bbox": [320, 1000, 325, 56], "text": "11.06.1822", "IdNr": 38737, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1822#1"}
{"file": "g01_22.jpg", "bbox": [320, 1290, 1080, 130], "text": "Julia Schönlein Strauß", "IdNr": 38314, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1822#2"}
{"file": "g01_22.jpg", "bbox": [400, 1230, 300, 60], "text": "28.05.1822", "IdNr": 38314, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1822#2"}
{"file": "g01_22.jpg", "bbox": [760, 1635, 250, 53], "text": "Wolf Strauß", "IdNr": 38813, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1822#3"}
{"file": "g01_22.jpg", "bbox": [420, 1543, 347, 55], "text": "29.07.1822", "IdNr": 38813, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1822#3"}
{"file": "g01_22.jpg", "bbox": [660, 1853, 340, 62], "text": "Vrommet Michel", "IdNr": 38742, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1822#4"}
{"file": "g01_22.jpg", "bbox": [420, 1773, 313, 55], "text": "29.11.1822", "IdNr": 38742, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1822#4"}
{"file": "g02_22.jpg", "bbox": [140, 543, 205, 118], "text": "Elias Mosbacher", "IdNr": 34023, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-n34023", "note": "given name 'Elias' legible, surname line faint/abbreviated in ink; label from index. TRIPLE dup index persons 34023/35707/38748 = same entry, labeled once (34023). Date pairing unique (25.06.1823); bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 703, 195, 95], "text": "Hajum Strauß", "IdNr": 37361, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-n37361", "note": "ink faint; index 'Strauss, Hajum (Chaim Sohn des David Halevi)'; date pairing 18.08.1823; bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 913, 180, 87], "text": "Jakob Michel", "IdNr": 38015, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38015", "note": "small mark line below name (possible death-note start, d.18.07.1825 per index) excluded from bbox; bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1073, 180, 90], "text": "Kindle Michel", "IdNr": 38018, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38018", "note": "bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1273, 180, 110], "text": "Vromeln Michel", "IdNr": 38022, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-n38022", "note": "given name reads 'Vromeln'?/'Vromme'?; index 'Michel, NN, Vromeln'; date pairing 27.02.1825; bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1453, 180, 100], "text": "Abraham Strauß", "IdNr": 38784, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38784", "note": "bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1603, 180, 95], "text": "Vromel Michel", "IdNr": 38738, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38738", "note": "bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1813, 180, 120], "text": "Isaak Strasburger", "IdNr": 38031, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38031", "note": "bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [1840, 538, 405, 115], "text": "den 25ten Juni 1823, 11 Uhr", "IdNr": 34023, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d34023", "note": "hour qualifier word illegible - omitted; no weekday words in this register; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 683, 405, 100], "text": "den 18ten August 1823, Morgens", "IdNr": 37361, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d37361", "note": "day numeral ambiguous in ink (reads '19ten'?; index 18.08.1823, label from index); hour numeral illegible - omitted; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 893, 405, 80], "text": "den 3ten October 1824, 11 Uhr", "IdNr": 38015, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38015", "note": "hour qualifier word illegible - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1078, 405, 115], "text": "den 1ten November 1824, Morgens", "IdNr": 38018, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38018", "note": "hour numeral weak - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1273, 405, 80], "text": "den 27ten Februar 1825, Mittags 1 Uhr", "IdNr": 38022, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38022", "note": "month word weak; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1448, 405, 125], "text": "den 18ten November 1825, Morgens", "IdNr": 38784, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38784", "note": "hour numeral illegible - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1595, 405, 158], "text": "den 24ten Juni 1826, Morgens", "IdNr": 38738, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38738", "note": "hour numeral weak - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1822, 405, 156], "text": "den 7ten August 1826, 5 Uhr", "IdNr": 38031, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38031", "note": "hour qualifier word faint - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g04_22.jpg", "bbox": [150, 365, 280, 135], "text": "Veila Mosbacher", "IdNr": 38037, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38037", "note": "ink reads 'Veile', index 'Veila'; label from index"}
{"file": "g04_22.jpg", "bbox": [150, 540, 280, 142], "text": "Mendel Mosbacher", "IdNr": 3763, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n3763", "note": "duplicate index person 3764 = same entry, labeled once (3763); the g12/g13 father as a CHILD here (b.18.10.1831)"}
{"file": "g04_22.jpg", "bbox": [150, 741, 280, 148], "text": "Jakob Michel", "IdNr": 38043, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38043"}
{"file": "g04_22.jpg", "bbox": [150, 1255, 280, 145], "text": "Moses Strauß", "IdNr": 38794, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-n38794", "note": "surname heavy flourish, semi-legible"}
{"file": "g04_22.jpg", "bbox": [150, 1450, 280, 115], "text": "Joseph Mosbacher", "IdNr": 6403, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6403", "note": "duplicate index person 38095 = same entry, labeled once (6403); later-hand death note '+ d. 9 Aug. 1835' below name EXCLUDED from bbox"}
{"file": "g04_22.jpg", "bbox": [150, 1700, 280, 120], "text": "Isaak Strauß", "IdNr": 38097, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38097", "note": "later-hand small repeat annotation below excluded from bbox"}
{"file": "g04_22.jpg", "bbox": [150, 1880, 280, 137], "text": "Madel Mosbacher", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38098", "note": "d.08.04.1837 per index (also s03 death register person)"}
{"file": "g04_22.jpg", "bbox": [2100, 370, 400, 146], "text": "23.04.1831", "IdNr": 38037, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38037", "note": "'d 23t Apr: 1831 Morgens 1 Uhr' legible; no weekday words in this register"}
{"file": "g04_22.jpg", "bbox": [2100, 541, 400, 139], "text": "18.10.1831", "IdNr": 3763, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d3763", "note": "'d 18t Oct: 1831 Morgens 8 Uhr' legible"}
{"file": "g04_22.jpg", "bbox": [2100, 712, 400, 130], "text": "19.12.1831", "IdNr": 38043, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38043", "note": "'d 19t Dec 1831 Mittags 4 Uhr' - day numeral weak, month/year legible"}
{"file": "g04_22.jpg", "bbox": [2100, 1148, 400, 96], "text": "14.06.1833", "IdNr": 38794, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38794", "note": "'d 14ten Jun 1833 Abends 4 Uhr' legible"}
{"file": "g04_22.jpg", "bbox": [2100, 1252, 400, 152], "text": "18.08.1833", "IdNr": 6403, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6403", "note": "'d 18t Aug 1833 Abends 6(?) Uhr' legible"}
{"file": "g04_22.jpg", "bbox": [2100, 1660, 400, 200], "text": "15.01.1835", "IdNr": 38097, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38097", "note": "'d 15t Jan ...5 Abends 6 Uhr' - year tail weak"}
{"file": "g04_22.jpg", "bbox": [2100, 1924, 400, 176], "text": "28.11.1835", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38098", "note": "'d 28t Nov 1835 Abends 9 Uhr' legible"}
{"file": "g05_22.jpg", "bbox": [160, 171, 240, 382], "text": "Rebecca Mosbacher", "IdNr": 38099, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.0-1"}
{"file": "g05_22.jpg", "bbox": [2050, 170, 420, 200], "text": "10.01.1836", "IdNr": 38099, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.1-2"}
{"file": "g05_22.jpg", "bbox": [2050, 572, 420, 179], "text": "15.09.1837", "IdNr": 38100, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3"}
{"file": "g05_22.jpg", "bbox": [160, 744, 240, 191], "text": "Emanuel Mosbacher", "IdNr": 38101, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.3"}
{"file": "g05_22.jpg", "bbox": [2050, 751, 420, 179], "text": "07.04.1838", "IdNr": 38101, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.4"}
{"file": "g05_22.jpg", "bbox": [160, 935, 240, 191], "text": "Josef Leopold Mosbacher", "IdNr": 6407, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.4"}
{"file": "g05_22.jpg", "bbox": [2050, 930, 420, 179], "text": "08.04.1841", "IdNr": 6407, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.5"}
{"file": "g05_22.jpg", "bbox": [160, 1126, 240, 191], "text": "Therese Regensburger", "IdNr": 38103, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.5"}
{"file": "g05_22.jpg", "bbox": [2050, 1109, 420, 179], "text": "06.06.1841", "IdNr": 38103, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.6"}
{"file": "g05_22.jpg", "bbox": [160, 1317, 240, 191], "text": "Aaron Strauß", "IdNr": 38106, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "g05_22.jpg", "bbox": [2050, 1288, 420, 179], "text": "14.09.1841", "IdNr": 38106, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.7"}
{"file": "g05_22.jpg", "bbox": [160, 1508, 240, 191], "text": "Resel Mosbacher", "IdNr": 38107, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.7"}
{"file": "g05_22.jpg", "bbox": [2050, 1467, 420, 179], "text": "21.10.1841", "IdNr": 38107, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.8"}
{"file": "g05_22.jpg", "bbox": [160, 1699, 240, 191], "text": "Emanuel Mosbacher", "IdNr": 38113, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.8"}
{"file": "g05_22.jpg", "bbox": [2050, 1646, 420, 179], "text": "28.01.1843", "IdNr": 38113, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.9"}
{"file": "g05_22.jpg", "bbox": [160, 1890, 240, 191], "text": "Ernestine Regensburger", "IdNr": 38198, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.9"}
{"file": "g05_22.jpg", "bbox": [2050, 1825, 420, 179], "text": "28.01.1843", "IdNr": 38198, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.10"}
{"file": "g05_22.jpg", "bbox": [160, 2025, 240, 162], "text": "Abraham Mosbacher", "IdNr": 38199, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.10"}
{"file": "g05_22.jpg", "bbox": [2050, 1950, 420, 225], "text": "29.02.1844", "IdNr": 38199, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.11"}
{"file": "g06_22.jpg", "bbox": [160, 219, 240, 183], "text": "Sara Strauß", "IdNr": 38671, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.1"}
{"file": "g06_22.jpg", "bbox": [2050, 230, 420, 190], "text": "02.05.1844", "IdNr": 38671, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.1"}
{"file": "g06_22.jpg", "bbox": [160, 402, 240, 183], "text": "Isaak Mosbacher", "IdNr": 38309, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.2"}
{"file": "g06_22.jpg", "bbox": [2050, 420, 420, 190], "text": "09.01.1845", "IdNr": 38309, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.2"}
{"file": "g06_22.jpg", "bbox": [160, 585, 240, 183], "text": "David Regensburger", "IdNr": 38763, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.3"}
{"file": "g06_22.jpg", "bbox": [2050, 610, 420, 190], "text": "22.02.1845", "IdNr": 38763, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3"}
{"file": "g06_22.jpg", "bbox": [160, 768, 240, 183], "text": "Koppel Strauß", "IdNr": 38814, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.4"}
{"file": "g06_22.jpg", "bbox": [2050, 800, 420, 190], "text": "02.05.1845", "IdNr": 38814, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.4"}
{"file": "g06_22.jpg", "bbox": [160, 951, 240, 183], "text": "Babeta Mosbacher", "IdNr": 38646, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.5"}
{"file": "g06_22.jpg", "bbox": [2050, 990, 420, 190], "text": "02.06.1846", "IdNr": 38646, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.5"}
{"file": "g06_22.jpg", "bbox": [160, 1134, 240, 183], "text": "Edel Strauß", "IdNr": 38796, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "g06_22.jpg", "bbox": [2050, 1180, 420, 190], "text": "11.11.1846", "IdNr": 38796, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.6"}
{"file": "g06_22.jpg", "bbox": [160, 1317, 240, 183], "text": "Abraham Regensburger", "IdNr": 38764, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.7"}
{"file": "g06_22.jpg", "bbox": [2050, 1370, 420, 190], "text": "03.02.1847", "IdNr": 38764, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.7"}
{"file": "g06_22.jpg", "bbox": [160, 1500, 240, 183], "text": "Daniel Strauß", "IdNr": 38815, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.8"}
{"file": "g06_22.jpg", "bbox": [2050, 1560, 420, 190], "text": "18.05.1847", "IdNr": 38815, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.8"}
{"file": "g06_22.jpg", "bbox": [160, 1683, 240, 183], "text": "Fanny Mosbacher", "IdNr": 23186, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.9"}
{"file": "g06_22.jpg", "bbox": [2050, 1750, 420, 190], "text": "21.06.1847", "IdNr": 23186, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.9"}
{"file": "g06_22.jpg", "bbox": [160, 1866, 240, 183], "text": "Leopold Rothschild", "IdNr": 38766, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.10"}
{"file": "g06_22.jpg", "bbox": [2050, 1940, 420, 190], "text": "23.06.1847", "IdNr": 38766, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.10"}
{"file": "g07_22.jpg", "bbox": [81, 690, 271, 190], "text": "Karoline Mosbacher", "IdNr": 38744, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.3-top"}
{"file": "g07_22.jpg", "bbox": [1908, 690, 348, 190], "text": "12.07.1849", "IdNr": 38744, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "0.3-top"}
{"file": "g08_22.jpg", "bbox": [150, 270, 355, 65], "text": "Jette Michel", "IdNr": 38739, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38739"}
{"file": "g08_22.jpg", "bbox": [150, 415, 355, 55], "text": "Fanny Strauß", "IdNr": 38785, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38785"}
{"file": "g08_22.jpg", "bbox": [150, 530, 355, 90], "text": "Gittel Mosbacher", "IdNr": 38745, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38745", "note": "index name 'Mosbacher, Gitta,Getta (Gittel Gattin des Menachem)'; label matches ink 'Gittel'"}
{"file": "g08_22.jpg", "bbox": [150, 750, 355, 150], "text": "Moses Regensburger", "IdNr": 38664, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38664", "note": "scribe's struck false-start line above included in bbox; ink order 'Regensburger / Moses'; person also appears on s05"}
{"file": "g08_22.jpg", "bbox": [150, 985, 355, 130], "text": "Gidel Mosbacher", "IdNr": 38751, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38751", "note": "first line struck through by scribe (false start reading '...Strauss...'); label 'Gidel Mosbacher'"}
{"file": "g08_22.jpg", "bbox": [150, 1205, 355, 90], "text": "Elias Michel", "IdNr": 38740, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38740"}
{"file": "g08_22.jpg", "bbox": [150, 1525, 355, 95], "text": "Mendel Mosbacher", "IdNr": 38756, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38756"}
{"file": "g08_22.jpg", "bbox": [150, 1760, 355, 110], "text": "Lazarus Strauß", "IdNr": 38799, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38799"}
{"file": "g08_22.jpg", "bbox": [2100, 268, 560, 120], "text": "den 7ten März 1852, früh 8 Uhr", "IdNr": 38739, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38739", "note": "weekday word illegible, omitted from label (no single-reader guesses); 07.03.1852 was a Sunday; numerals from index; 'März' and 'früh 8 Uhr' legible"}
{"file": "g08_22.jpg", "bbox": [2100, 398, 560, 100], "text": "Freitag den 8ten Mai 1852, Mittags 3 Uhr", "IdNr": 38785, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38785", "note": "weekday 'Freitag' legible but 08.05.1852 was a Saturday - scribe error, documented; numerals from index"}
{"file": "g08_22.jpg", "bbox": [2100, 542, 560, 130], "text": "Montag den 21. Juni 1852, Mittags 4 Uhr", "IdNr": 38745, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38745", "note": "weekday cross-check PASS (21.06.1852 = Monday)"}
{"file": "g08_22.jpg", "bbox": [2100, 668, 560, 88], "text": "Dienstag den 30ten November 1852, Mittags 2 Uhr", "IdNr": 38750, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38750", "note": "weekday cross-check PASS (30.11.1852 = Tuesday); NAME CELL EXCLUDED: near-duplicate of held-out eval name 'Gerson Mosbacher' (33739); flagged for parent ruling"}
{"file": "g08_22.jpg", "bbox": [2100, 798, 560, 135], "text": "Am Sonnabend den 27ten May 1854, Morgens 5 Uhr", "IdNr": 38664, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38664", "note": "weekday cross-check PASS (27.05.1854 = Saturday); year '1854' on separate line"}
{"file": "g08_22.jpg", "bbox": [2100, 1042, 560, 150], "text": "am Freitag d. 18ten August 1854, Morgens 4 Uhr", "IdNr": 38751, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38751", "note": "weekday cross-check PASS (18.08.1854 = Friday)"}
{"file": "g08_22.jpg", "bbox": [2100, 1208, 560, 195], "text": "am Donnerstag d. 28ten September 1854, Morgens 4 Uhr", "IdNr": 38740, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38740", "note": "weekday cross-check PASS (28.09.1854 = Thursday)"}
{"file": "g08_22.jpg", "bbox": [2100, 1448, 560, 315], "text": "Freitag den 29ten December 1854, Vormittags 10 Uhr", "IdNr": 38756, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38756", "note": "certificate header line ('...No 10') inside bbox, not transcribed; hour qualifier faint (reads 'Vormittags'?); weekday+date cross-check PASS (29.12.1854 = Friday)"}
{"file": "g08_22.jpg", "bbox": [2100, 1762, 560, 140], "text": "am Mittwoch d. 30ten May 1855, Morgens", "IdNr": 38799, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38799", "note": "hour numeral illegible, omitted from label; weekday cross-check PASS (30.05.1855 = Wednesday)"}
{"file": "g09_22.jpg", "bbox": [252, 840, 145, 125], "text": "Gina Michel", "IdNr": 38741, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38741", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; two-line 'Gina / Michel'"}
{"file": "g09_22.jpg", "bbox": [250, 1250, 130, 70], "text": "David", "IdNr": 38786, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38786", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; given name only as inked; surname Strauss in shared parents brace (cert N.104), twins with 38787"}
{"file": "g09_22.jpg", "bbox": [245, 1340, 175, 70], "text": "Jeanette", "IdNr": 38787, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38787", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; given name only as inked; surname Strauss in shared parents brace, twins with 38786"}
{"file": "g09_22.jpg", "bbox": [210, 1595, 280, 100], "text": "Maier Joseph Mosbacher", "IdNr": 38746, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38746", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; two-line 'Maier Joseph / Mosbacher'"}
{"file": "g09_22.jpg", "bbox": [210, 1838, 180, 52], "text": "Babette", "IdNr": 38800, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38800", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; given name only as inked ('posthuma' entry); surname Strauss from index/parent phrase"}
{"file": "g09_22.jpg", "bbox": [1900, 855, 700, 155], "text": "12.10.1856", "IdNr": 38741, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38741", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; no weekday word written; 'Am 12ten ... October 1856 ... 6 Uhr' read; 12.10.1856 = Sunday (no cross-check possible)"}
{"file": "g09_22.jpg", "bbox": [1900, 1215, 700, 240], "text": "20.10.1856", "IdNr": 38786, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38786", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; SHARED date cell with twin 38787; 'Am Montag den zwanzigsten October 1856' - Montag legible, 20.10.1856 = Monday PASS"}
{"file": "g09_22.jpg", "bbox": [1900, 1215, 700, 240], "text": "20.10.1856", "IdNr": 38787, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38787", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; SHARED date cell with twin 38786; 'Am Montag den zwanzigsten October 1856' - Montag legible, 20.10.1856 = Monday PASS"}
{"file": "g09_22.jpg", "bbox": [1900, 1610, 700, 190], "text": "09.03.1857", "IdNr": 38746, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38746", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; 'Am Montag ... Maerz 1857, Nachts 12 Uhr' - Montag legible, 09.03.1857 = Monday PASS; day numeral from index, pinned by weekday+month+year"}
{"file": "g09_22.jpg", "bbox": [1900, 1825, 750, 125], "text": "08.11.1857", "IdNr": 38800, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38800", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; 'Am Sonntag d. 8ten Nov. 1857 Abends' - Sonntag legible, 08.11.1857 = Sunday PASS"}
{"file": "g10_22.jpg", "bbox": [140, 350, 330, 110], "text": "Isaak Mosbacher", "IdNr": 38752, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38752"}
{"file": "g10_22.jpg", "bbox": [140, 560, 330, 95], "text": "Mina Strauß", "IdNr": 38675, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38675", "note": "index 'Strauß, Wilhelmine, Wilhelmine (Mina)'; ink reads 'Mina Strauß' (call name documented in index)"}
{"file": "g10_22.jpg", "bbox": [140, 845, 330, 95], "text": "Jendele Marx", "IdNr": 3664, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n3664", "note": "index 'Marx, Jette, Jendele'; ink reads 'Jendele Marx'; DUP index person 35455 = same entry, labeled once; small-print naming note below excluded from bbox"}
{"file": "g10_22.jpg", "bbox": [140, 1015, 330, 85], "text": "Manuel Mosbacher", "IdNr": 38674, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38674", "note": "death note '+ d. 6 Jan. 1860' below excluded from bbox"}
{"file": "g10_22.jpg", "bbox": [140, 1315, 330, 95], "text": "Elieser Marx", "IdNr": 38733, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38733"}
{"file": "g10_22.jpg", "bbox": [140, 1495, 330, 90], "text": "Mendel Strauß", "IdNr": 38788, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38788"}
{"file": "g10_22.jpg", "bbox": [140, 1610, 330, 90], "text": "Abraham Mosbacher", "IdNr": 38753, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38753"}
{"file": "g10_22.jpg", "bbox": [140, 1862, 330, 78], "text": "Veist Strauß", "IdNr": 38677, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38677", "note": "index 'Strauß, Veist (Faist, Feist)'; death note '+ 31 Marz 1862' below excluded from bbox"}
{"file": "g10_22.jpg", "bbox": [1745, 382, 325, 240], "text": "den 17ten Januar 1858, 10 Uhr", "IdNr": 38752, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38752", "note": "weekday word illegible - omitted (17.01.1858 was a Sunday); hour qualifier faint; numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 693, 325, 137], "text": "Am Mittwoch den 21ten July 1858", "IdNr": 38675, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38675", "note": "weekday cross-check PASS (21.07.1858 = Wednesday)"}
{"file": "g10_22.jpg", "bbox": [1745, 853, 325, 160], "text": "Am Mittwoch den 15ten September 1858, Morgens 4 Uhr", "IdNr": 3664, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d3664", "note": "weekday cross-check PASS (15.09.1858 = Wednesday)"}
{"file": "g10_22.jpg", "bbox": [1745, 1038, 325, 185], "text": "den 19ten December 1859, Mittags 11 Uhr", "IdNr": 38674, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38674", "note": "weekday word illegible - omitted (19.12.1859 was a Monday); numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 1293, 325, 155], "text": "den 4ten Februar 1860, Morgens 2 Uhr", "IdNr": 38733, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38733", "note": "weekday word illegible - omitted (04.02.1860 was a Saturday); ink word order '...2 Uhr / Morgens'; numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 1483, 325, 145], "text": "den 4ten April 1860, Morgens 3 Uhr", "IdNr": 38788, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38788", "note": "weekday word illegible - omitted (04.04.1860 was a Wednesday); ink word order '...3 Uhr / Morgens'; numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 1640, 325, 115], "text": "den 8ten December 1860, Nachmittags 3 Uhr", "IdNr": 38753, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38753", "note": "weekday word illegible - omitted (08.12.1860 was a Saturday); hour qualifier faint ('Nachmittags'?); numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 1862, 325, 140], "text": "den 30ten Januar 1862, 6 Uhr", "IdNr": 38677, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38677", "note": "weekday and hour-qualifier words faint - omitted (30.01.1862 was a Thursday); numerals from index"}
{"file": "g11_22.jpg", "bbox": [225, 460, 220, 110], "text": "Sara Strauß", "IdNr": 38789, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r1", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [235, 720, 200, 95], "text": "Felix Mosbacher", "IdNr": 3766, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r2", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [235, 890, 300, 140], "text": "Jeanette Strauß", "IdNr": 38806, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r3", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [210, 1200, 230, 135], "text": "Louis Mosbacher", "IdNr": 6446, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r4", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [185, 1475, 220, 135], "text": "David Strauß", "IdNr": 38807, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r5", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [210, 1620, 250, 130], "text": "Amalie Mosbacher", "IdNr": 38757, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r6", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [195, 1825, 290, 90], "text": "Jeanette Strauß", "IdNr": 38694, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-r7", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [230, 1935, 240, 100], "text": "Fanny Strauß", "IdNr": 38808, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r8", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 490, 420, 210], "text": "15.09.1863", "IdNr": 38789, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1863", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 735, 420, 150], "text": "18.03.1864", "IdNr": 3766, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1864", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 955, 420, 240], "text": "30.08.1865", "IdNr": 38806, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1865", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1210, 420, 165], "text": "14.09.1866", "IdNr": 6446, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1866", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1505, 420, 150], "text": "06.07.1867", "IdNr": 38807, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1867", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1780, 420, 105], "text": "28.01.1869", "IdNr": 38757, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-1869a", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1880, 420, 95], "text": "25.01.1869", "IdNr": 38694, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-1869b", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1960, 420, 140], "text": "12.02.1869", "IdNr": 38808, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-1869c", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 495, 390, 105], "text": "Siegmund Strauß", "IdNr": 38697, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38697", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 670, 390, 95], "text": "Siegmund Strauß", "IdNr": 38790, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38790", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 795, 390, 95], "text": "Isaak Strauß", "IdNr": 38699, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38699", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 905, 390, 95], "text": "Rickchen Mosbacher", "IdNr": 6449, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6449", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1065, 390, 100], "text": "Abraham Strauß", "IdNr": 38781, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38781", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1185, 390, 115], "text": "Jeanette Strauß", "IdNr": 38791, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38791", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1380, 390, 90], "text": "Sara Strauß", "IdNr": 38698, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38698", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1615, 390, 85], "text": "Zerline Mosbacher", "IdNr": 6500, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6500", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1735, 390, 90], "text": "Regine Strauß", "IdNr": 38809, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38809", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1850, 390, 135], "text": "Lina Mosbacher", "IdNr": 6412, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6412", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [1560, 555, 900, 125], "text": "15.03.1870", "IdNr": 38697, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38697", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Dienstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 690, 900, 125], "text": "05.04.1870", "IdNr": 38790, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38790", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday word illegible; numeral 5 + April clear, matches index"}
{"file": "g12_22.jpg", "bbox": [1560, 830, 900, 120], "text": "04.06.1870", "IdNr": 38699, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38699", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Samstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 950, 900, 120], "text": "02.06.1870", "IdNr": 6449, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6449", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Donnerstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 1140, 900, 115], "text": "24.03.1871", "IdNr": 38781, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38781", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday reads 'Sonntag' but 24.03.1871=Friday - clerk error or misread; numeral 24 + März clear, matches index"}
{"file": "g12_22.jpg", "bbox": [1560, 1255, 900, 145], "text": "21.09.1871", "IdNr": 38791, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38791", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday semi-legible (best read Donnerstag, matches calendar Thursday); numeral 21 + September clear, matches index"}
{"file": "g12_22.jpg", "bbox": [1560, 1421, 900, 139], "text": "14.05.1872", "IdNr": 38698, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38698", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Dienstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 1614, 900, 115], "text": "15.08.1872", "IdNr": 6412, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6412", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Donnerstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 1743, 900, 102], "text": "05.10.1872", "IdNr": 38809, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38809", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Samstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 1865, 900, 130], "text": "18.06.1872", "IdNr": 6500, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6500", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; cell reads 'Dienstag den 8ten[struck] achtzehnten Juni' + 1897 margin correction confirming 18; Dienstag OK for 18.06.1872"}
{"file": "g13_22.jpg", "bbox": [230, 630, 350, 140], "text": "Jeanette Strauß", "IdNr": 38783, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38783", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 815, 350, 115], "text": "Jeta Mosbacher", "IdNr": 38754, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38754", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1020, 350, 120], "text": "Moses Strauß", "IdNr": 38792, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38792", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1195, 350, 110], "text": "Ruben Strauß", "IdNr": 38810, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38810", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1365, 350, 125], "text": "Selma Strauß", "IdNr": 38782, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38782", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1520, 350, 125], "text": "Felix Mosbacher", "IdNr": 38755, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38755", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1670, 350, 115], "text": "Friederike Strauß", "IdNr": 38793, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38793", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [1520, 700, 940, 140], "text": "21.10.1873", "IdNr": 38783, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38783", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Dienstag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 840, 940, 140], "text": "21.12.1873", "IdNr": 38754, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38754", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Sonntag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1100, 940, 130], "text": "06.01.1874", "IdNr": 38792, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38792", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Dienstag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1250, 940, 100], "text": "26.02.1874", "IdNr": 38810, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38810", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Donnerstag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1455, 940, 110], "text": "07.05.1875", "IdNr": 38782, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38782", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Freitag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1575, 940, 110], "text": "25.08.1875", "IdNr": 38755, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38755", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Mittwoch OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1695, 940, 110], "text": "19.09.1875", "IdNr": 38793, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38793", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Sonntag OK"}
{"file": "s01_22.jpg", "bbox": [373, 425, 118, 72], "text": "Strauß", "IdNr": 38312, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "s01-n1", "pairing": "narrative", "pairing_note": "narrative death entry 1821; full name line-broken (Kop-pel Strauß), surname word labeled; date not in index, unlabeled per single-reader rule"}
{"file": "s02_22.jpg", "bbox": [157, 495, 235, 164], "text": "Giedle Mosbacher", "IdNr": 38315, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R3@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 495, 474, 164], "text": "28.06.1823", "IdNr": 38315, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R3"}
{"file": "s02_22.jpg", "bbox": [157, 659, 235, 164], "text": "Salomon Mosbacher", "IdNr": 38622, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R4@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 659, 474, 164], "text": "04.10.1823", "IdNr": 38622, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R4"}
{"file": "s02_22.jpg", "bbox": [157, 823, 235, 164], "text": "Riefke Rothschild", "IdNr": 38623, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R5@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 823, 474, 164], "text": "09.04.1825", "IdNr": 38623, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R5"}
{"file": "s02_22.jpg", "bbox": [157, 987, 235, 164], "text": "Löw Rothschild", "IdNr": 38624, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R6@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 987, 474, 164], "text": "11.05.1825", "IdNr": 38624, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R6"}
{"file": "s02_22.jpg", "bbox": [157, 1315, 235, 164], "text": "Ruchme Mosbacher", "IdNr": 38625, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "R8@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 1315, 474, 164], "text": "08.04.1827", "IdNr": 38625, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R8"}
{"file": "s02_22.jpg", "bbox": [157, 1479, 235, 164], "text": "Gundle Mosbacher", "IdNr": 38626, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "R9@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 1479, 474, 164], "text": "07.12.1827", "IdNr": 38626, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R9"}
{"file": "s02_22.jpg", "bbox": [157, 1807, 235, 164], "text": "Lorenz Mosbacher", "IdNr": 38627, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R11@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 1807, 474, 164], "text": "29.03.1829", "IdNr": 38627, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R11"}
{"file": "s02_22.jpg", "bbox": [157, 1971, 235, 164], "text": "Michael Michel", "IdNr": 38628, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R12@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 1971, 474, 164], "text": "21.12.1829", "IdNr": 38628, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R12"}
{"file": "s03_22.jpg", "bbox": [147, 526, 226, 164], "text": "Gudle Straus", "IdNr": 38633, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.3"}
{"file": "s03_22.jpg", "bbox": [1292, 501, 442, 167], "text": "26.11.1831", "IdNr": 38633, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3"}
{"file": "s03_22.jpg", "bbox": [147, 1018, 226, 164], "text": "Jakob Mehnl", "IdNr": 38636, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "s03_22.jpg", "bbox": [1292, 1002, 442, 167], "text": "22.05.1834", "IdNr": 38636, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.6"}
{"file": "s03_22.jpg", "bbox": [147, 1510, 226, 164], "text": "Jonas Mosbacher", "IdNr": 38639, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.9"}
{"file": "s03_22.jpg", "bbox": [1292, 1503, 442, 167], "text": "09.08.1835", "IdNr": 38639, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.9"}
{"file": "s03_22.jpg", "bbox": [147, 1674, 226, 164], "text": "Michelle Michael", "IdNr": 38640, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.10"}
{"file": "s03_22.jpg", "bbox": [1292, 1670, 442, 167], "text": "26.06.1836", "IdNr": 38640, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.10"}
{"file": "s03_22.jpg", "bbox": [147, 1838, 226, 164], "text": "Madel Mosbacher", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.11"}
{"file": "s03_22.jpg", "bbox": [1292, 1837, 442, 167], "text": "08.04.1837", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.11"}
{"file": "s05_22.jpg", "bbox": [150, 190, 390, 105], "text": "Isaak Michel", "IdNr": 35242, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 155, 370, 135], "text": "06.05.1850", "IdNr": 35242, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 385, 390, 65], "text": "Jette Marx", "IdNr": 38649, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 295, 370, 170], "text": "22.05.1850", "IdNr": 38649, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 490, 390, 80], "text": "Jette Michel", "IdNr": 38651, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 485, 370, 140], "text": "22.07.1850", "IdNr": 38651, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 635, 390, 80], "text": "Eva Rothschild", "IdNr": 38652, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 635, 370, 117], "text": "20.10.1850", "IdNr": 38652, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 720, 390, 115], "text": "Mina Mosbacher", "IdNr": 38653, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 755, 370, 130], "text": "28.04.1853", "IdNr": 38653, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 945, 390, 80], "text": "Simson Michel", "IdNr": 38654, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 950, 370, 155], "text": "08.09.1853", "IdNr": 38654, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 1055, 390, 130], "text": "Vaibel Mosbacher", "IdNr": 38041, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 1150, 370, 185], "text": "30.12.1853", "IdNr": 38041, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 1330, 390, 105], "text": "David Strauß", "IdNr": 38655, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 1340, 370, 140], "text": "20.07.1854", "IdNr": 38655, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 1445, 390, 125], "text": "Moses Regensburger", "IdNr": 38664, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 1490, 370, 135], "text": "13.12.1854", "IdNr": 38664, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 1595, 390, 110], "text": "Mendel Mosbacher", "IdNr": 38665, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 1650, 370, 185], "text": "29.12.1854", "IdNr": 38665, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s06_22.jpg", "bbox": [140, 134, 420, 220], "text": "Kehle Mosbacher", "IdNr": 38667, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.0"}
{"file": "s06_22.jpg", "bbox": [1290, 250, 240, 178], "text": "28.01.1856", "IdNr": 38667, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.0-1"}
{"file": "s06_22.jpg", "bbox": [1290, 428, 240, 149], "text": "14.04.1856", "IdNr": 38668, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.2"}
{"file": "s06_22.jpg", "bbox": [140, 574, 420, 220], "text": "Jeanette Strauß", "IdNr": 38669, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.2"}
{"file": "s06_22.jpg", "bbox": [1290, 577, 240, 149], "text": "26.10.1856", "IdNr": 38669, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3"}
{"file": "s06_22.jpg", "bbox": [140, 794, 420, 220], "text": "Alexander Strauß", "IdNr": 38035, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.3"}
{"file": "s06_22.jpg", "bbox": [1290, 900, 240, 124], "text": "19.10.1857", "IdNr": 38035, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.5"}
{"file": "s06_22.jpg", "bbox": [140, 1234, 420, 220], "text": "Sara Strauß", "IdNr": 38671, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.5"}
{"file": "s06_22.jpg", "bbox": [1290, 1090, 240, 170], "text": "19.03.1858", "IdNr": 38671, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.6-7"}
{"file": "s06_22.jpg", "bbox": [140, 1454, 420, 220], "text": "Seligmann Hirsch", "IdNr": 38672, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "s06_22.jpg", "bbox": [1290, 1322, 240, 149], "text": "11.08.1858", "IdNr": 38672, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.8"}
{"file": "s06_22.jpg", "bbox": [140, 1674, 420, 220], "text": "Mattche Marx", "IdNr": 38673, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.7"}
{"file": "s06_22.jpg", "bbox": [1290, 1620, 240, 149], "text": "27.02.1859", "IdNr": 38673, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.10"}
{"file": "s06_22.jpg", "bbox": [140, 1894, 420, 220], "text": "Manuel Mosbacher", "IdNr": 38674, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.8"}
{"file": "s06_22.jpg", "bbox": [1290, 1910, 270, 230], "text": "06.01.1860", "IdNr": 38674, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.11-12"}
{"file": "s07_22.jpg", "bbox": [150, 142, 430, 160], "text": "Rachel Strauß", "IdNr": 38676, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.0"}
{"file": "s07_22.jpg", "bbox": [1300, 145, 320, 155], "text": "31.05.1860", "IdNr": 38676, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.0"}
{"file": "s07_22.jpg", "bbox": [150, 302, 430, 160], "text": "Wilhelmine Strauß", "IdNr": 38675, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.1"}
{"file": "s07_22.jpg", "bbox": [1300, 430, 320, 180], "text": "24.03.1861", "IdNr": 38675, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.1-2"}
{"file": "s07_22.jpg", "bbox": [150, 622, 430, 160], "text": "Veist Strauß", "IdNr": 38677, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.3"}
{"file": "s07_22.jpg", "bbox": [1300, 615, 320, 160], "text": "31.03.1862", "IdNr": 38677, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3-4"}
{"file": "s07_22.jpg", "bbox": [150, 1262, 430, 160], "text": "Jeanette Strauß", "IdNr": 38036, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.7"}
{"file": "s07_22.jpg", "bbox": [1300, 1225, 320, 185], "text": "22.11.1864", "IdNr": 38036, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.7"}
{"file": "s07_22.jpg", "bbox": [150, 1742, 430, 160], "text": "Jeanette Strauß", "IdNr": 38694, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.10"}
{"file": "s07_22.jpg", "bbox": [1300, 1685, 320, 160], "text": "08.02.1869", "IdNr": 38694, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.10"}
{"file": "s07_22.jpg", "bbox": [150, 1902, 430, 160], "text": "Jeanette Strauß", "IdNr": 38695, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.11"}
{"file": "s07_22.jpg", "bbox": [1300, 1845, 320, 160], "text": "11.02.1869", "IdNr": 38695, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.11"}
{"file": "s08_22.jpg", "bbox": [150, 500, 410, 165], "text": "Siegmund Strauß", "IdNr": 38697, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "2"}
{"file": "s08_22.jpg", "bbox": [1300, 490, 870, 190], "text": "31.05.1870", "IdNr": 38697, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "2"}
{"file": "s08_22.jpg", "bbox": [150, 705, 410, 110], "text": "Charlotte Mosbacher", "IdNr": 38042, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "3"}
{"file": "s08_22.jpg", "bbox": [1300, 695, 870, 185], "text": "08.07.1871", "IdNr": 38042, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "3"}
{"file": "s08_22.jpg", "bbox": [150, 925, 410, 150], "text": "Sara Strauß", "IdNr": 38698, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "4"}
{"file": "s08_22.jpg", "bbox": [1300, 900, 870, 215], "text": "05.06.1872", "IdNr": 38698, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "4"}
{"file": "s08_22.jpg", "bbox": [150, 1225, 410, 180], "text": "Clara Mosbacher", "IdNr": 38039, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "5"}
{"file": "s08_22.jpg", "bbox": [1300, 1180, 870, 235], "text": "02.04.1874", "IdNr": 38039, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "5"}
{"file": "s08_22.jpg", "bbox": [150, 1430, 410, 145], "text": "Isaak Strauß", "IdNr": 38699, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "6"}
{"file": "s08_22.jpg", "bbox": [1300, 1425, 870, 135], "text": "15.11.1875", "IdNr": 38699, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "6"}
{"file": "g07_22.jpg", "bbox": [81, 150, 271, 181], "text": "Regina Mosbacher", "IdNr": 38749, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.1"}
{"file": "g07_22.jpg", "bbox": [81, 331, 271, 334], "text": "Josef Rothschild", "IdNr": 1945, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.2"}
{"file": "g07_22.jpg", "bbox": [81, 1096, 271, 174], "text": "Madel Strauß", "IdNr": 38816, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.4"}
{"file": "g07_22.jpg", "bbox": [81, 1407, 271, 199], "text": "Mina Mosbacher", "IdNr": 38653, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "g07_22.jpg", "bbox": [81, 1606, 271, 138], "text": "Sarah Rothschild", "IdNr": 38780, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.7"}
{"file": "g07_22.jpg", "bbox": [1908, 162, 348, 161], "text": "25.03.1848", "IdNr": 38749, "project": "eschau1081-batch1", "kind": "date", "conf": "medium", "band": "1.1"}
{"file": "g07_22.jpg", "bbox": [1908, 323, 348, 333], "text": "25.01.1849", "IdNr": 1945, "project": "eschau1081-batch1", "kind": "date", "conf": "medium", "band": "1.2"}
{"file": "g07_22.jpg", "bbox": [1908, 1086, 348, 173], "text": "25.10.1849", "IdNr": 38816, "project": "eschau1081-batch1", "kind": "date", "conf": "medium", "band": "1.5"}
{"file": "g07_22.jpg", "bbox": [1908, 1394, 348, 200], "text": "28.03.1851", "IdNr": 38653, "project": "eschau1081-batch1", "kind": "date", "conf": "verified", "band": "1.7"}
{"file": "g07_22.jpg", "bbox": [1908, 1594, 348, 137], "text": "08.04.1851", "IdNr": 38780, "project": "eschau1081-batch1", "kind": "date", "conf": "verified", "band": "1.8"}
{"file": "g07_22.jpg", "bbox": [1908, 1731, 348, 156], "text": "05.01.1852", "IdNr": 38765, "project": "eschau1081-batch1", "kind": "date", "conf": "verified", "band": "1.9"}
{"file": "g07_22.jpg", "bbox": [1908, 1887, 348, 253], "text": "08.02.1852", "IdNr": 38798, "project": "eschau1081-batch1", "kind": "date", "conf": "verified", "band": "1.10"}
{"file": "s04_22.jpg", "bbox": [150, 92, 280, 232], "text": "Rachel Strauß", "project": "eschau-seeds", "kind": "name", "IdNr": 38641}
{"file": "s04_22.jpg", "bbox": [150, 296, 280, 165], "text": "Raphael Simon", "project": "eschau-seeds", "kind": "name", "IdNr": 38642}
{"file": "s04_22.jpg", "bbox": [890, 322, 360, 157], "text": "30.03.1840", "project": "eschau-seeds", "kind": "date", "IdNr": 38642}
{"file": "s04_22.jpg", "bbox": [150, 605, 280, 173], "text": "Wolf Mehnl", "project": "eschau-seeds", "kind": "name", "IdNr": 38644}
{"file": "s04_22.jpg", "bbox": [890, 631, 360, 165], "text": "14.04.1840", "project": "eschau-seeds", "kind": "date", "IdNr": 38644}
{"file": "s04_22.jpg", "bbox": [150, 433, 280, 200], "text": "Gudle Straus", "project": "eschau-seeds", "kind": "name", "IdNr": 38643}
{"file": "s04_22.jpg", "bbox": [890, 459, 360, 192], "text": "17.04.1840", "project": "eschau-seeds", "kind": "date", "IdNr": 38643}
{"file": "s04_22.jpg", "bbox": [150, 750, 280, 195], "text": "Isaak Mosbacher", "project": "eschau-seeds", "kind": "name", "IdNr": 38309}
{"file": "s04_22.jpg", "bbox": [890, 776, 360, 173], "text": "19.01.1845", "project": "eschau-seeds", "kind": "date", "IdNr": 38309, "conf": "medium", "note": "faint/partially illegible ink; band+row verified 2026-11-06; numerals index-anchored"}
{"file": "s04_22.jpg", "bbox": [150, 1309, 280, 176], "text": "Babeta Mosbacher", "project": "eschau-seeds", "kind": "name", "IdNr": 38646}
{"file": "s04_22.jpg", "bbox": [890, 1335, 360, 168], "text": "14.06.1846", "project": "eschau-seeds", "kind": "date", "IdNr": 38646}
{"file": "s04_22.jpg", "bbox": [150, 1457, 280, 208], "text": "Jakob Solinger", "project": "eschau-seeds", "kind": "name", "IdNr": 38647}
{"file": "s04_22.jpg", "bbox": [890, 1483, 360, 200], "text": "03.02.1848", "project": "eschau-seeds", "kind": "date", "IdNr": 38647}
{"file": "s04_22.jpg", "bbox": [150, 1637, 280, 178], "text": "Sara Rothschild", "project": "eschau-seeds", "kind": "name", "IdNr": 38648}
{"file": "s04_22.jpg", "bbox": [890, 1663, 360, 170], "text": "13.04.1848", "project": "eschau-seeds", "kind": "date", "IdNr": 38648}
{"file": "s04_22.jpg", "bbox": [150, 1787, 280, 189], "text": "Berez Löb Mosbacher", "project": "eschau-seeds", "kind": "name", "IdNr": 6400}
{"file": "s04_22.jpg", "bbox": [890, 1813, 360, 181], "text": "20.04.1849", "project": "eschau-seeds", "kind": "date", "IdNr": 6400, "conf": "medium", "note": "faint ink; band verified 2026-11-06; numerals index-anchored"}
{"file": "s04_22.jpg", "bbox": [150, 1948, 280, 158], "text": "Jette Michel", "project": "eschau-seeds", "kind": "name", "IdNr": 35241}
{"file": "s04_22.jpg", "bbox": [890, 1974, 360, 150], "text": "04.05.1849", "project": "eschau-seeds", "kind": "date", "IdNr": 35241, "conf": "medium", "note": "very faint ink; band verified 2026-11-06; numerals index-anchored"}
{"file": "s02_22.jpg", "bbox": [180, 1108, 220, 259], "text": "Jakob Michel", "project": "eschau-seeds", "kind": "name", "IdNr": 38015}
{"file": "s04_22.jpg", "bbox": [434, 296, 166, 165], "text": "Lehrer, Vorsänger", "project": "eschau-occ", "kind": "occupation", "IdNr": 38642}
{"file": "s04_22.jpg", "bbox": [434, 1457, 166, 208], "text": "Handelsmann", "project": "eschau-occ", "kind": "occupation", "IdNr": 38647}
{"file": "s04_22.jpg", "bbox": [434, 1787, 166, 189], "text": "Händler", "project": "eschau-occ", "kind": "occupation", "IdNr": 6400}
{"file": "s05_22.jpg", "bbox": [474, 173, 152, 201], "text": "Handelsmann", "project": "eschau-occ", "kind": "occupation", "IdNr": 35242}
{"file": "s05_22.jpg", "bbox": [458, 1036, 166, 138], "text": "Krämer, Ökonom", "project": "eschau-occ", "kind": "occupation", "IdNr": 38041}
{"file": "s05_22.jpg", "bbox": [458, 1254, 166, 178], "text": "Viehhändler", "project": "eschau-occ", "kind": "occupation", "IdNr": 38655}
{"file": "s05_22.jpg", "bbox": [458, 1630, 166, 106], "text": "Händler", "project": "eschau-occ", "kind": "occupation", "IdNr": 38665}
{"file": "s06_22.jpg", "bbox": [450, 1671, 166, 266], "text": "Schreibmaterialienhändler", "project": "eschau-occ", "kind": "occupation", "IdNr": 38673}
{"file": "g10_22.jpg", "bbox": [1745, 448, 330, 128], "text": "d. 17ten siebzehnten Jan. 1858", "IdNr": 38752, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
{"file": "g10_22.jpg", "bbox": [1745, 872, 330, 80], "text": "den Mittwoch 15ten fünfzehnten", "IdNr": 3664, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
{"file": "g07_22.jpg", "bbox": [1880, 545, 270, 105], "text": "fünfundzwanzigsten Jan. 1849", "IdNr": 38779, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
{"file": "g07_22.jpg", "bbox": [1880, 1465, 280, 130], "text": "Freitag den 28. achtundzwanzigsten März 1851", "IdNr": 38653, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
{"file": "g07_22.jpg", "bbox": [1900, 1770, 280, 115], "text": "Montag den 5. fünften Jan. 1852", "IdNr": 38765, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
'''
_cells = [json.loads(l) for l in _ARMB_BLOB.strip().splitlines()]
print(f"merged v15 target cells: {len(_cells)} ({_merged_path})", flush=True)
_drop = [c for c in _cells if str(c.get("IdNr")) in _held_idnrs]
if _drop:
    print(f"held-out decontamination: dropping {len(_drop)} cells for persons {sorted({str(c['IdNr']) for c in _drop})}", flush=True)
    _cells = [c for c in _cells if str(c.get("IdNr")) not in _held_idnrs]
_gm = [c for c in _cells if str(c.get("text", "")).strip() == "Gerson Mosbacher"]
if _gm:
    print(f"text-level held-out leak: dropping {len(_gm)} cells carrying eval-target text", flush=True)
    _cells = [c for c in _cells if str(c.get("text", "")).strip() != "Gerson Mosbacher"]
if not SMOKE:
    assert len(_cells) == int(os.environ.get("KURRENT_EXPECT_CELLS", "306")), f"expected 306 merged arm-B cells, got {len(_cells)}"
_train_path = CODE / "train_manifest_v4.jsonl"
print("train manifest:", _train_path.name, flush=True)
samples = [json.loads(l) for l in _train_path.open()]
_mm = [s2 for s2 in samples if s2.get("project") not in ("eschau-seeds", "eschau-occ", "eschau-gbirths")]
REPEAT = 4  # v19: v15/v18 weight (v16 x6 regressed names 0.8894->0.9276)  # v12 (issue #32): was 40 in v8-v10; x40 drove memorization (seeds_fit 0.0, held-out names worse than base)
samples = _mm + _cells * REPEAT
_rnd.seed(SEED); _rnd.shuffle(samples)
print(f"mix: MM lines {len(_mm)} + target {len(_cells)}x{REPEAT} = {len(samples)}", flush=True)
samples = [s for s in samples if 2 <= len(s.get("text", "").strip()) <= 480]
print(f"train line-samples: {len(samples)}", flush=True)

# --- sanity (2026-10-19 v2: seeds present, MM_1_006 excluded, eval pages resolve) ---
from collections import Counter as _Counter
_proj = _Counter(s.get("project", "?") for s in samples)
print("DATA dir:", str(DATA), flush=True)
print("project counts:", dict(_proj), flush=True)
# _held_idnrs is defined at the embedded-cells load above (v14); assert below still guards batch-1/batch-2 cells
_leaked = sorted({str(s.get("IdNr")) for s in samples if str(s.get("IdNr")) in _held_idnrs})
if not SMOKE:
    assert "MM_1_006" not in _proj, "MM_1_006 leaked into training manifest!"
    assert not _leaked, f"held-out IdNrs leaked into training: {_leaked}"
_eschau = sum(1 for s in samples if s.get("project") == "eschau-seeds")
print(f"ESCHAU seed lines: {_eschau}", flush=True)
_seed_files = sorted(set(s["file"] for s in samples if s.get("project") == "eschau-seeds"))
print("ESCHAU seed pages:", _seed_files, flush=True)
_missing = sorted(f for f in set(s["file"] for s in samples) if f not in PAGE_ROOT)
print("missing pages:", _missing if _missing else "none", flush=True)
_eval_rows = EVAL500_ROWS
_eval_missing = sorted(set(r["file"] for r in _eval_rows) - set(PAGE_ROOT))
print("eval500 rows:", len(_eval_rows), "eval pages missing:", _eval_missing if _eval_missing else "none", flush=True)
if not SMOKE:
    assert len(_eval_rows) == 500, f"eval set should be 500 rows, got {len(_eval_rows)}"
    assert not _missing, f"training pages missing: {_missing[:5]}"
    assert not _eval_missing, f"eval pages missing: {_eval_missing[:5]}"
progress_push(progress_payload("boot", seed=SEED, train_samples=len(samples), target_cells=len(_cells),
                               missing_pages=len(_missing), eval_pages_missing=len(_eval_missing)))
if _seed_files:
    _im = Image.open(PAGE_ROOT[_seed_files[0]])
    print("seed page resolution:", _seed_files[0], _im.size, flush=True)
    if not SMOKE:
        assert _im.size[0] > 1000, "seed page resolution too small - wrong image?"

# Group line indices by page, then iterate pages in a fixed permutation so each
# page image is decoded once per worker per epoch.
by_page = {}
for idx, s in enumerate(samples):
    by_page.setdefault(s["file"], []).append(idx)
pages = sorted(by_page)
rng = random.Random(42)
rng.shuffle(pages)
order = [i for p in pages for i in by_page[p]]
print(f"train pages: {len(pages)}", flush=True)


class LineDataset(torch.utils.data.Dataset):
    def __init__(self):
        self._cache = {}

    def __len__(self):
        return len(order)

    def __getitem__(self, i):
        s = samples[order[i]]
        crop = crop_from_page(s)
        pixels = processor(images=crop, return_tensors="pt").pixel_values[0]
        labels = processor.tokenizer(s["text"], return_tensors="pt", truncation=True, max_length=500).input_ids[0]
        return {"pixel_values": pixels, "labels": labels}


def collate(batch):
    pixels = torch.stack([b["pixel_values"] for b in batch])
    maxlen = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : len(b["labels"])] = b["labels"]
    return {"pixel_values": pixels, "labels": labels}


BEST = {"score": basenames.get("cer", float("inf")) if "basenames" in dir() else float("inf"), "step": 0}  # v18: init baseline - a trained ckpt must beat step-0 names CER (v17 had no baseline and picked a names-collapsed step)
DRIVE_BEST = os.environ.get("KURRENT_DRIVE_BEST", "")  # e.g. /content/drive/MyDrive/kurrent-colab-results/v3-checkpoint-best


def combined_score(r12: dict, rnames: dict) -> float:
    # FROZEN 2026-10-23: keep-best = held-out Eschau EVAL_NAMES CER only (validation set).
    if rnames and rnames.get("n"):
        return rnames["cer"]
    return score12(r12)


def drive_save_best():
    """Atomic-ish copy of checkpoint-best to Drive: build tmp dir, then rename."""
    if not DRIVE_BEST:
        return
    import shutil
    tmp = DRIVE_BEST + ".tmp"
    try:
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(str(OUT / "checkpoint-best"), tmp)
        if os.path.exists(DRIVE_BEST):
            shutil.rmtree(DRIVE_BEST)
        os.rename(tmp, DRIVE_BEST)
        print("DRIVE checkpoint-best updated", flush=True)
    except Exception as e:  # Drive hiccup must not kill training
        print(f"DRIVE best-save failed: {type(e).__name__}: {e}", flush=True)


class KeepBestCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step > 0 and state.global_step % EVAL_EVERY == 0:
            try:
                r = evaluate12(f"step{state.global_step}")
                rn = evaluate_names(f"step{state.global_step}")
                s = combined_score(r, rn)
                if s < BEST["score"]:
                    BEST.update(score=s, step=state.global_step)
                    trainer.save_model(str(OUT / "checkpoint-best"))
                    print(f"NEW BEST step={state.global_step} namesCER={s:.3f} -> checkpoint-best saved", flush=True)
                    drive_save_best()
                progress_push(progress_payload(
                    f"step{state.global_step}", step=state.global_step,
                    score12=round(score12(r), 4),
                    names_n=rn.get("n"), names_cer=rn.get("cer"), names_word_acc=rn.get("word_acc"),
                    names_cer_sel=round(s, 4), names_v4subset_cer=rn.get("cer_v4subset"),
                    names_span_hit_rate=rn.get("span_hit_rate"),
                    best_step=BEST["step"], best_score=BEST["score"]))
            except Exception as e:  # never let keep-best kill a run
                print(f"keep-best eval failed at step {state.global_step}: {type(e).__name__}: {e}", flush=True)
            finally:
                model.train()


args = Seq2SeqTrainingArguments(
    output_dir=str(OUT / "ckpt"),
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS if MAX_STEPS else -1,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    lr_scheduler_type="linear",
    warmup_steps=WARMUP,
    seed=SEED,
    data_seed=SEED,
    fp16=DEVICE == "cuda",
    save_strategy="no",
    logging_steps=10,
    report_to=[],
    dataloader_num_workers=2,
    remove_unused_columns=False,
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=LineDataset(), data_collator=collate, callbacks=[KeepBestCallback()])
print("STAGE train-start", flush=True)
t_train = time.time()
trainer.train()
train_secs = time.time() - t_train
print(f"train_seconds={train_secs:.0f}", flush=True)
trainer.save_model(str(OUT / "checkpoint-final"))

print("STAGE eval-after-final", flush=True)
final12 = evaluate12("after-final")

print("STAGE eval-best", flush=True)
if BEST["step"] > 0:
    try:
        model = VisionEncoderDecoderModel.from_pretrained(str(OUT / "checkpoint-best")).to(DEVICE)
        best12 = evaluate12(f"best-step{BEST['step']}")
        best500 = evaluate500(f"best-step{BEST['step']}")
        bestnames = evaluate_names(f"best-step{BEST['step']}")
    except Exception as e:  # fall back to scoring the final model
        print(f"best-checkpoint load failed: {type(e).__name__}: {e} - scoring final model", flush=True)
        model = trainer.model
        best12 = evaluate12("best-load-failed-final")
        best500 = evaluate500("best-load-failed-final")
        bestnames = evaluate_names("best-load-failed-final")
else:
    print("no improvement over base during training - best = base", flush=True)
    best12, best500, bestnames = evaluate12("best-is-base"), base500, basenames

# seeds-fit diagnostic: can the model reproduce TRAINED seed cells? (fit-check, not generalization)
def evaluate_seeds_fit():
    import random as _rnd
    _rnd.seed(7)
    seed_rows = [s for s in samples if s.get("project") == "eschau-seeds"]
    pick = _rnd.sample(seed_rows, min(12, len(seed_rows)))
    model.eval()
    cers = []
    for r in pick:
        hyp = read_pil(crop_from_page(r))
        cers.append(cer(hyp, r["text"]))
    out = {"n": len(pick), "cer": round(sum(cers) / max(1, len(cers)), 4)}
    print(f"EVAL_SEEDS_FIT: {json.dumps(out, ensure_ascii=False)}", flush=True)
    return out

seeds_fit = evaluate_seeds_fit()

summary = {
    "seed": SEED,
    "config": {"lr": LR, "warmup": WARMUP, "epochs": EPOCHS, "eval_every": EVAL_EVERY, "batch": BATCH, "base_model": CKPT_SRC, "resume": True, "left_out": "MM_1_006"},
    "before": {"h12": base12, "mm006_500": base500, "names": basenames},
    "after_final": {"h12": final12},
    "best": {"step": BEST["step"], "names_cer": round(BEST["score"], 4) if BEST["step"] > 0 else None, "h12": best12, "mm006_500": best500, "names": bestnames},
    "train_seconds": round(train_secs),
    "total_seconds": round(time.time() - T0),
    "samples": len(samples),
    "seeds_fit": seeds_fit,
    "pages": len(pages),
    "max_steps": MAX_STEPS,
}
(OUT / "timing_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
progress_push(progress_payload("train-done", seed=SEED, result=str(OUT / "timing_result.json")))
print("STAGE done", json.dumps(summary["best"]["mm006_500"]), flush=True)
