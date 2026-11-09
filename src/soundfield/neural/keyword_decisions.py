"""Resolve keyword detections against competing phrases and independent speech evidence."""
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..contracts import audio_array, nonnegative_integer, positive_integer


@dataclass(frozen=True)
class KeywordEvent:
    sample: int
    phrase: str


def detect_keyword_events(audio, sample_rate_hz, detector, *, normalize_level=False):
    """Decode a 16 kHz mono recording continuously, including the final partial block."""
    wave = audio_array(audio)
    if sample_rate_hz != 16000 or wave.shape[1] != 1:
        raise ValueError("keyword decoding requires one 16 kHz channel")
    wave = wave[:,0]
    if not len(wave):
        return []
    if normalize_level:
        padded = np.pad(wave, (0, -len(wave)%160))
        energy = np.mean(padded.reshape(-1,160)**2, axis=1)
        recent_peak = maximum_filter1d(energy, 40, origin=19, mode="constant")
        gain = np.minimum(12., .08/np.maximum(np.sqrt(recent_peak),1e-8))
        wave = np.clip(wave*np.repeat(gain,160)[:len(wave)], -.95, .95)
    stream = detector.create_stream()
    events = []

    def decode(available):
        while detector.is_ready(stream):
            detector.decode_stream(stream)
            phrase = detector.get_result(stream)
            if phrase:
                events.append(KeywordEvent(available,phrase))
                detector.reset_stream(stream)

    for start in range(0,len(wave),1600):
        block = wave[start:start+1600]
        stream.accept_waveform(sample_rate_hz,block.astype(np.float32))
        decode(start+len(block))
    stream.accept_waveform(sample_rate_hz,np.zeros(8000,np.float32))
    stream.input_finished()
    decode(len(wave))
    return events


def keyword_text_evidence(phrase, transcript, *, competing=(), supporting=()):
    """Classify full phrases and configured distinguishing fragments of a transcript."""
    if phrase.isascii():
        text = " "+" ".join(transcript.upper().split())+" "
        contains = lambda term: " "+term.upper()+" " in text
        enough = len(text.split()) >= len(phrase.split())
    else:
        text = "".join(transcript.split())
        contains = lambda term: term in text
        enough = len(text) >= len(phrase)
    if contains(phrase):
        return "match"
    if any(term and contains(term) for term in competing):
        return "conflict"
    if any(term and contains(term) for term in supporting):
        return "support"
    return "other" if enough else "uncertain"


def _in_window(samples, start, stop):
    index = bisect_left(samples,start)
    return index < len(samples) and samples[index] <= stop


def confirm_keyword_events(primary, contextual, sample_rate_hz, sample_count, *, confusable_phrases, verify):
    """Combine searches with a 200 ms confirmation interval.

    Contextual distractor labels map to the trigger they resemble. The verifier
    receives the trigger and microphone sample bounds. It returns 'match',
    'support', 'conflict', 'other' or 'uncertain'. A positive primary event survives an
    inconclusive observation. Output timestamps include confirmation time.
    """
    positive_integer(sample_rate_hz, "sample_rate_hz")
    nonnegative_integer(sample_count, "sample_count")
    window = round(.7*sample_rate_hz)
    lookahead = round(.2*sample_rate_hz)
    original_times, distractor_times = defaultdict(list), defaultdict(list)
    events = sorted([(h,True) for h in primary]+[(h,False) for h in contextual],
                    key=lambda item:(item[0].sample,not item[1]))
    for hit, original in events:
        nonnegative_integer(hit.sample,"event sample")
        if hit.sample > sample_count or not hit.phrase:
            raise ValueError("keyword events require a phrase and a sample inside the recording")
        if original:
            original_times[hit.phrase].append(hit.sample)
        elif hit.phrase in confusable_phrases:
            distractor_times[confusable_phrases[hit.phrase]].append(hit.sample)
    sensitive = set(confusable_phrases.values())
    last, output = {}, []
    for hit, original in events:
        if hit.phrase in confusable_phrases or hit.sample-last.get(hit.phrase,-window-1)<=window:
            continue
        ready = min(hit.sample+lookahead,sample_count)
        primary_support = original or _in_window(original_times[hit.phrase],hit.sample-window,ready)
        if not primary_support or hit.phrase in sensitive:
            verdict = verify(hit.phrase,max(0,hit.sample-3*sample_rate_hz),ready)
            if verdict not in ("match","support","conflict","other","uncertain"):
                raise ValueError("unsupported keyword verification result")
            disputed = _in_window(distractor_times[hit.phrase],hit.sample-window,ready)
            if verdict=="conflict" or (not primary_support and verdict!="match") or (disputed and verdict=="other"):
                continue
        last[hit.phrase] = hit.sample
        output.append(KeywordEvent(ready,hit.phrase))
    return output
