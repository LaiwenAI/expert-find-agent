"""Frozen official Qwen3 Embedding using MLX last-real-token pooling."""
from pathlib import Path

TASK='Retrieve scholarly publications documenting research relevant to the submitted paper.'


class QwenEmbedding:
    def __init__(self,path='artifacts/models/qwen3-embedding-0.6b',max_length=512,batch_size=8):
        import mlx.core as mx
        from mlx_lm.models.qwen3 import Model,ModelArgs
        from transformers import AutoTokenizer
        import json
        if type(max_length) is not int or max_length<1 or type(batch_size) is not int or batch_size<1:raise ValueError('Invalid encoding budget')
        if not (Path(path)/'checkpoint-provenance.json').exists():raise ValueError('Verify official frozen checkpoint first')
        mx.set_memory_limit(4*1024**3);mx.set_cache_limit(128*1024**2)
        config=json.loads((Path(path)/'config.json').read_text())
        self.model=Model(ModelArgs.from_dict(config))
        # Embedding checkpoint stores bare backbone keys, unlike the LM wrapper.
        self.model.model.load_weights(list(mx.load(str(Path(path)/'model.safetensors')).items()),strict=True)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.mx=mx
        self.max_length=max_length;self.batch_size=batch_size
        self.pad_id=self.tokenizer.pad_token_id
        if self.pad_id is None:self.pad_id=self.tokenizer.eos_token_id

    def encode(self,texts,query=False):
        import numpy as np
        mx=self.mx
        inputs=[('Instruct: '+TASK+'\nQuery:'+text) if query else text for text in texts]
        encoded=[self.tokenizer.encode(text,add_special_tokens=True)[:self.max_length] for text in inputs]
        if any(not ids for ids in encoded):raise ValueError('Empty token sequence')
        order=sorted(range(len(encoded)),key=lambda i:(len(encoded[i]),i));result=[None]*len(encoded)
        for start in range(0,len(order),self.batch_size):
            selected=order[start:start+self.batch_size];lengths=[len(encoded[i]) for i in selected];width=max(lengths)
            ids=mx.array([encoded[i]+[self.pad_id]*(width-len(encoded[i])) for i in selected])
            hidden=self.model.model(ids)
            vectors=hidden[mx.arange(len(selected)),mx.array(lengths)-1].astype(mx.float32)
            vectors=vectors/mx.sqrt(mx.sum(vectors*vectors,axis=-1,keepdims=True));mx.eval(vectors)
            array=np.asarray(vectors,dtype=np.float32)
            if not np.isfinite(array).all():raise ValueError('Nonfinite embedding')
            for i,vector in zip(selected,array):result[i]=vector
        return np.asarray(result,dtype=np.float32)
