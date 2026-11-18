"""Compare source navigation within fixed expert candidates on public CoF SIGIR."""
import json
import statistics
import time
from pathlib import Path

import numpy as np

from experiments.retrieval.cof_centroid_bound import digest, split_sources
from expert_search.cof_benchmark import load


def dots(matrix, query):
    return np.einsum('ij,j->i', matrix, query, optimize=False)


def main(output):
    root, _, reviewers, _ = load('SIGIR')
    manifest = Path('artifacts/cof-gpu-public-input-v1/input-provenance.json')
    provenance = json.loads(manifest.read_text())
    for path, expected in provenance['input_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Changed frozen input: {path}')
    query_ids = provenance['query_ids']
    expert_ids = sorted(reviewers)
    paper_ids = sorted({p for expert in reviewers.values() for p in expert['papers']})
    position = {paper: i for i, paper in enumerate(paper_ids)}
    profiles = [np.array(sorted({position[p] for p in reviewers[e]['papers']})) for e in expert_ids]
    vectors = np.load(root/'qwen-embedding-torch-v1/vectors.npy', allow_pickle=False).astype(np.float64)
    if vectors.shape != (len(paper_ids) + len(query_ids), 1024):
        raise ValueError('Unexpected vector layout')
    source_vectors = vectors[:len(paper_ids)]

    trees = []
    for profile in profiles:
        groups = split_sources(source_vectors, profile, 4)
        centers = np.stack([source_vectors[group].mean(axis=0) for group in groups])
        radii = np.array([np.linalg.norm(source_vectors[group] - center, axis=1).max()
                          for group, center in zip(groups, centers)])
        trees.append((groups, centers, radii))

    def run(method, candidates, query):
        started = time.perf_counter_ns()
        seen = {}
        centroid_count = 0
        rankings = []
        query_norm = np.linalg.norm(query) if method == 'bounded' else None
        for expert in candidates:
            profile = profiles[expert]
            if method == 'flat':
                groups = [profile]
                group_order = [0]
                bounds = None
            else:
                groups, centers, radii = trees[expert]
                scores = dots(centers, query)
                centroid_count += len(centers)
                if method == 'bounded':
                    bounds = scores + radii * query_norm + 1e-10
                    group_order = np.argsort(-bounds, kind='stable')
                else:
                    bounds = None
                    group_order = [int(np.argmax(scores))]
            selected = []
            target = min(3, len(profile))
            for group in group_order:
                if bounds is not None and len(selected) == target and bounds[group] < selected[-1][1]:
                    break
                unseen = [p for p in groups[group] if p not in seen]
                if unseen:
                    seen.update(zip(unseen, dots(source_vectors[unseen], query)))
                selected = sorted(selected + [(p, seen[p]) for p in groups[group]],
                                  key=lambda hit: (-hit[1], hit[0]))[:3]
            rankings.append([int(p) for p, _ in selected])
        return {'rankings': rankings, 'source_scores': len(seen),
                'centroid_scores': centroid_count, 'milliseconds': (time.perf_counter_ns() - started) / 1e6}

    methods = ('flat', 'bounded', 'one_group')
    results = []
    cohorts = []
    for i, query_id in enumerate(query_ids):
        query = vectors[len(paper_ids) + i]
        all_scores = dots(source_vectors, query)
        candidates = sorted(range(len(profiles)), key=lambda e: (-all_scores[profiles[e]].max(), e))[:5]
        cohorts.append((query_id, query, candidates))
    for method in methods:
        run(method, cohorts[0][2], cohorts[0][1])
    for i, (query_id, query, candidates) in enumerate(cohorts):
        row = {'query_id': query_id, 'arms': {}}
        for repeat in range(5):
            offset = (i + repeat) % len(methods)
            for method in methods[offset:] + methods[:offset]:
                row['arms'].setdefault(method, []).append(run(method, candidates, query))
        reference = row['arms']['flat'][0]['rankings']
        for method in methods:
            for arm in row['arms'][method]:
                if method != 'one_group' and arm['rankings'] != reference:
                    raise ValueError(f'Different exact source order for {query_id}')
                arm['ordered_matches'] = sum(a == b for a, b in zip(arm['rankings'], reference))
                arm['reference_retained'] = sum(len(set(a) & set(b)) for a, b in zip(arm['rankings'], reference))
                arm['reference_sources'] = sum(map(len, reference))
        results.append(row)

    output.mkdir(exist_ok=False)
    (output/'rows.json').write_text(json.dumps(results, indent=2)+'\n')
    report = {'queries': len(results), 'candidates_per_query': 5, 'repeats_per_query': 5,
              'input_provenance_sha256': digest(manifest), 'rows_sha256': digest(output/'rows.json'),
              'code_sha256': {p.name: digest(p) for p in (Path(__file__), Path('experiments/retrieval/cof_centroid_bound.py'))},
              'arms': {}, 'scope': 'Reused public CoF SIGIR, cached FP64 vectors and frozen exact dense candidates. Single-process warm CPU timing excludes tree build, query encoding, candidate generation, paper text reads and LLM calls. Source retention measures overlap with the exact similarity reference, not semantic evidence support.'}
    for method in methods:
        rows = [row['arms'][method][0] for row in results]
        paired_means = [statistics.mean(r['milliseconds'] for r in row['arms'][method]) for row in results]
        report['arms'][method] = {
            'mean_source_scores': statistics.mean(r['source_scores'] for r in rows),
            'mean_centroid_scores': statistics.mean(r['centroid_scores'] for r in rows),
            'fraction_reference_sources_retained': sum(r['reference_retained'] for r in rows) / sum(r['reference_sources'] for r in rows),
            'ordered_matches': sum(r['ordered_matches'] for r in rows),
            'mean_milliseconds': statistics.mean(paired_means),
        }
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
