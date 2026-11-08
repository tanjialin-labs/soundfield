"""Evaluate a declared collection of source and room-response pairs."""

import argparse
import json
from pathlib import Path

from soundfield.dereverb import WpeConfig
from soundfield.io import read_audio, write_audio, load_impulse_array
from soundfield.pipelines import evaluate_reverberation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output_directory', type=Path)
    arguments = parser.parse_args()
    cases = json.loads(arguments.manifest.read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases:
        parser.error('manifest must contain a nonempty list of cases')
    identifiers, resources = set(), {arguments.manifest.resolve()}
    prepared = []
    for case in cases:
        identifier = case['name']
        if not isinstance(identifier, str) or not identifier.isascii() or not identifier.replace('_', '').replace('-', '').isalnum() or identifier in identifiers:
            parser.error('case names must be distinct ASCII letters, digits, underscores or hyphens')
        identifiers.add(identifier)
        source_path = (arguments.manifest.parent / case['reference']).resolve()
        response_path = (arguments.manifest.parent / case['response']).resolve()
        resources.update((source_path, response_path))
        rate, clean = read_audio(source_path)
        if response_path.suffix.lower() in ('.mat', '.npz'):
            response_rate, response = load_impulse_array(response_path, **case['response_array'])
        else:
            response_rate, response = read_audio(response_path)
        if rate != response_rate:
            parser.error('source and response sample rates must match')
        prepared.append((case, rate, clean, response))
    output = arguments.output_directory.resolve()
    targets = {output / 'scores.json'} | {output / name / filename for name in identifiers for filename in ('reverberant.wav', 'enhanced.wav')}
    if resources & targets:
        parser.error('outputs must differ from all input resources')
    results = []
    for case, rate, clean, response in prepared:
        parameters = dict(case.get('prediction', {}))
        parameters.setdefault('channel_count', max(clean.shape[1], response.shape[1]))
        config = WpeConfig(**parameters)
        result = evaluate_reverberation(clean, response, rate, prediction_config=config,
            response_gain=case.get('response_gain', 4.),
            metrics=case.get('metrics', ('spectral_distance', 'coherent_diffuse', 'modulation', 'perceptual')))
        case_output = output / case['name']
        case_output.mkdir(parents=True, exist_ok=True)
        write_audio(case_output / 'reverberant.wav', result.reverberant_audio, rate, encoding='float64')
        write_audio(case_output / 'enhanced.wav', result.enhanced_audio, rate, encoding='float64')
        results.append(dict(name=case['name'], sample_rate_hz=rate, delay_samples=result.delay_samples,
                            input=result.input_scores, output=result.output_scores))
    (output / 'scores.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
