"""Detect registered keywords in a 16 kHz mono recording using two search vocabularies."""
import argparse
import json
from pathlib import Path

import numpy as np

from soundfield.io import read_audio
from soundfield.neural.keyword_decisions import confirm_keyword_events, detect_keyword_events, keyword_text_evidence
from soundfield.neural.recognition_input import prepare_recognition_segment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("microphone",type=Path)
    parser.add_argument("output",type=Path)
    for name in ("tokens","encoder","decoder","joiner","keywords","context-keywords","phrase-map","ctc-model","ctc-tokens"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--enhancer",type=Path)
    args=parser.parse_args()
    resources=[p for k,p in vars(args).items() if k!="output" and isinstance(p,Path)]
    if any(p.resolve()==args.output.resolve() for p in resources):
        parser.error("output must differ from every input resource")
    import sherpa_onnx
    from pocketsphinx import Decoder
    from soundfield.neural.recurrent_spectrum import RecurrentSpectrumEnhancer

    rate,mic=read_audio(args.microphone)
    if rate!=16000 or mic.shape[1]!=1:
        parser.error("keyword detection requires mono audio at 16 kHz")
    audio=RecurrentSpectrumEnhancer(args.enhancer).enhance_audio(mic,rate) if args.enhancer else mic
    options=dict(tokens=str(args.tokens),encoder=str(args.encoder),decoder=str(args.decoder),
                 joiner=str(args.joiner),num_threads=1,keywords_threshold=.25)
    primary=sherpa_onnx.KeywordSpotter(**options,keywords_file=str(args.keywords))
    contextual=sherpa_onnx.KeywordSpotter(**options,keywords_file=str(args.context_keywords),num_trailing_blanks=0)
    registered=detect_keyword_events(audio,rate,primary)
    alternatives=detect_keyword_events(audio,rate,contextual,normalize_level=True)
    mapping=json.loads(args.phrase_map.read_text())
    chinese=sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(tokens=str(args.ctc_tokens),model=str(args.ctc_model),num_threads=1)
    english=Decoder(samprate=rate,loglevel="ERROR")

    def verify(phrase,start,stop):
        signal=prepare_recognition_segment(mic[start:stop],mic[start:stop])[:,0]
        if phrase.isascii():
            english.start_utt()
            english.process_raw(np.rint(np.clip(signal,-1,32767/32768)*32768).astype("<i2").tobytes(),False,True)
            english.end_utt()
            result=english.hyp()
            words=result.hypstr.upper().split() if result else []
            text=" ".join(words)
        else:
            stream=chinese.create_stream()
            stream.accept_waveform(rate,np.r_[signal,np.zeros(8000)].astype(np.float32))
            stream.input_finished()
            while chinese.is_ready(stream):
                chinese.decode_stream(stream)
            text=chinese.get_result(stream).replace(" ","")
        return keyword_text_evidence(phrase,text,competing=mapping["competing_text"].get(phrase,[]),
                                     supporting=mapping.get("support_text",{}).get(phrase,[]))

    events=confirm_keyword_events(registered,alternatives,rate,len(mic),
                                  confusable_phrases=mapping["distractors"],verify=verify)
    args.output.write_text(json.dumps([vars(e) for e in events],ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__":
    main()
