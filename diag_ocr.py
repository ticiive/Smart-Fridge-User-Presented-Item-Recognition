"""Diagnostico do Vision OCR: testa variantes e mostra qual funciona."""
import sys, os, tempfile, traceback
import cv2

img = sys.argv[1] if len(sys.argv) > 1 else "capturas/20260923_160439/passagem_007_a.jpg"
arr_color = cv2.imread(img)
arr_gray  = cv2.cvtColor(arr_color, cv2.COLOR_BGR2GRAY)

import Vision
from Foundation import NSURL

def run(arr, options, langs, level_accurate=True, label=""):
    fd, tmp = tempfile.mkstemp(suffix=".png"); os.close(fd)
    cv2.imwrite(tmp, arr)
    try:
        url = NSURL.fileURLWithPath_(os.path.abspath(tmp))
        req = Vision.VNRecognizeTextRequest.alloc().init()
        if level_accurate:
            req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        if langs is not None:
            req.setRecognitionLanguages_(langs)
        req.setUsesLanguageCorrection_(True)
        h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, options)
        ok, err = h.performRequests_error_([req], None)
        res = req.results()
        txt = " ".join(str(o.topCandidates_(1)[0].string()) for o in (res or []) if o.topCandidates_(1))
        print(f"[OK ] {label}: ok={ok} err={err} texto={txt[:80]!r}")
    except Exception:
        print(f"[ERR] {label}")
        traceback.print_exc(limit=3)
    finally:
        os.unlink(tmp)

print("pyobjc Vision:", Vision.__file__)
run(arr_color, {},   ["pt-BR","en-US"], True,  "color + options{} + pt/en")
run(arr_color, None, ["pt-BR","en-US"], True,  "color + optionsNone + pt/en")
run(arr_color, None, None,              True,  "color + optionsNone + sem idiomas")
run(arr_gray,  None, None,              True,  "gray  + optionsNone + sem idiomas")
run(arr_color, None, ["en-US"],         True,  "color + optionsNone + en")
