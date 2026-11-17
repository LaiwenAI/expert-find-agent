"""Build candidate-LM and candidate-first evidence indexes from a local corpus."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline,verify_evidence_documents
from expert_search.compiled_expert_search import corpus_snapshot
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.corpus import records
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.fast_retrieval import FastSearch,add_selective_indexes,compile_index


def build(corpus,source_database,language_index,evidence_database):
    corpus=Path(corpus);source_database=Path(source_database);language_index=Path(language_index);evidence_database=Path(evidence_database)
    if not source_database.is_file():raise FileNotFoundError('Build the source lexical index first')
    if language_index.exists() or evidence_database.exists():raise FileExistsError('Choose new index paths; existing indexes are not overwritten')
    experts={r['id'] for _,r in records(corpus/'experts.jsonl')};documents={};profiles=defaultdict(list)
    for _,row in records(corpus/'evidence.jsonl'):
        if row['id'] in documents or row['expert_id'] not in experts:raise ValueError('Duplicate evidence or unknown owner')
        documents[row['id']]=row['text'];profiles[row['expert_id']].append(row['id'])
    if not documents:raise ValueError('Empty evidence corpus')
    language_index.parent.mkdir(parents=True,exist_ok=True);evidence_database.parent.mkdir(parents=True,exist_ok=True)
    compile_index(source_database,evidence_database);add_selective_indexes(evidence_database)
    lookup=FastSearch(corpus,evidence_database);verify_evidence_documents(corpus,lookup);lookup.db.close()
    CompiledCandidateLM(ExpertLanguageModels(documents,profiles)).save(language_index,corpus_snapshot(corpus))
    engine=CandidateEvidencePipeline(corpus,language_index,evidence_database)
    engine.evidence.db.close()
    return {'documents':len(documents),'experts_with_evidence':len(profiles),'language_index':str(language_index),
            'evidence_database':str(evidence_database),'scope':'Offline construction; source score correctness requires separate audit. No relevance or precision validation.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--corpus',type=Path,default=Path('artifacts/corpus'))
    parser.add_argument('--source-database',type=Path,default=Path('artifacts/search.sqlite'))
    parser.add_argument('--language-index',type=Path,default=Path('artifacts/teacher-compiled-lm'))
    parser.add_argument('--evidence-database',type=Path,default=Path('artifacts/fast-search-selective-v1.sqlite'))
    args=parser.parse_args();print(json.dumps(build(args.corpus,args.source_database,args.language_index,args.evidence_database),indent=2))
