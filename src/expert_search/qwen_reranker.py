"""Frozen official Qwen3 reranker with MLX and a two-row FP32 output head.

This is a baseline implementation, not task training or a new scoring formula.
Right padding is safe for selected real positions in a causal transformer.
"""
import math
from pathlib import Path

SYSTEM='Decide whether the document is relevant to the query under the instruction. Answer yes or no.'
TASK='Find researchers whose documented research background is relevant to the submitted paper.'
PREFIX='<|im_start|>system\n'+SYSTEM+'<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def pack_tokens(prefix,body,suffix,max_length):
    if type(max_length) is not int or max_length<=len(prefix)+len(suffix):
        raise ValueError('Context must leave room for body tokens')
    raw=len(prefix)+len(body)+len(suffix)
    return list(prefix)+list(body[:max_length-len(prefix)-len(suffix)])+list(suffix),raw


class QwenReranker:
    def __init__(self,path='artifacts/models/qwen3-reranker-0.6b',max_length=4096,batch_size=4):
        import mlx.core as mx
        from mlx_lm import load
        if type(batch_size) is not int or batch_size<1:raise ValueError('Invalid batch size')
        if not (Path(path)/'checkpoint-provenance.json').exists():raise ValueError('Prepare pinned official checkpoint first')
        self.model,self.tokenizer=load(path,trust_remote_code=False)
        self.mx=mx;self.max_length=max_length;self.batch_size=batch_size
        self.prefix=self.tokenizer.encode(PREFIX,add_special_tokens=False)
        self.suffix=self.tokenizer.encode(SUFFIX,add_special_tokens=False)
        no=self.tokenizer.encode('no',add_special_tokens=False)
        yes=self.tokenizer.encode('yes',add_special_tokens=False)
        if len(no)!=1 or len(yes)!=1 or no==yes:raise ValueError('Expected distinct single-token yes/no')
        self.answer_ids=[no[0],yes[0]]
        self.weight=(self.model.model.embed_tokens.weight if self.model.args.tie_word_embeddings else self.model.lm_head.weight)
        self.answer_weight=self.weight[mx.array(self.answer_ids)].astype(mx.float32)
        mx.eval(self.answer_weight)
        self.pad_id=self.tokenizer.pad_token_id
        if self.pad_id is None:self.pad_id=self.tokenizer.eos_token_id
        self.dtype=str(self.weight.dtype)
        pack_tokens(self.prefix,[1],self.suffix,max_length)

    def encode(self,query,document):
        body=self.tokenizer.encode(f'<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {document}',add_special_tokens=False)
        return pack_tokens(self.prefix,body,self.suffix,self.max_length)

    def _batch(self,sequences):
        mx=self.mx
        lengths=[len(s) for s in sequences];width=max(lengths)
        inputs=mx.array([s+[self.pad_id]*(width-len(s)) for s in sequences])
        hidden=self.model.model(inputs)
        last=hidden[mx.arange(len(sequences)),mx.array(lengths)-1].astype(mx.float32)
        logits=last@self.answer_weight.T
        probability=mx.softmax(logits,axis=-1)[:,1]
        mx.eval(probability)
        return probability.tolist()

    def score(self,query,documents):
        encoded=[self.encode(query,d) for d in documents]
        order=sorted(range(len(encoded)),key=lambda i:(len(encoded[i][0]),i))
        scores=[None]*len(documents)
        for start in range(0,len(order),self.batch_size):
            selected=order[start:start+self.batch_size]
            values=self._batch([encoded[i][0] for i in selected])
            for i,value in zip(selected,values):scores[i]=float(value)
        if any(not math.isfinite(v) or not 0<=v<=1 for v in scores):raise ValueError('Invalid reranker score')
        return dict(scores=scores,input_tokens=[len(ids) for ids,_ in encoded],
            untruncated_tokens=[raw for _,raw in encoded],truncated=sum(raw>len(ids) for ids,raw in encoded))

    def validate_projection(self):
        """Compare against dense full-vocabulary FP32 projection on a short input."""
        mx=self.mx
        sequence,_=self.encode('robot motion control','Research on robot motion control and trajectory planning.')
        hidden=self.model.model(mx.array([sequence]))
        selected=hidden[:,-1,:].astype(mx.float32)@self.answer_weight.T
        full=hidden.astype(mx.float32)@self.weight.astype(mx.float32).T
        reference=full[:,-1,:][:,mx.array(self.answer_ids)]
        a,b=mx.softmax(selected,axis=-1),mx.softmax(reference,axis=-1)
        error=mx.max(mx.abs(a-b));mx.eval(error)
        if float(error)>2e-5:raise ValueError('Selected answer head differs from dense reference')
        return dict(maximum_probability_error=float(error),tolerance=2e-5,
            scope='Same BF16 transformer states; two-row versus dense all-position/all-vocabulary FP32 head. Not a PyTorch-versus-MLX equivalence claim.')
