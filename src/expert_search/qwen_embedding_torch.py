"""Official frozen Qwen3 Embedding backbone on a Slurm-allocated CUDA device."""
from pathlib import Path
import numpy as np
from .qwen_embedding import TASK


class QwenTorchEmbedding:
    def __init__(self,path='artifacts/models/qwen3-embedding-0.6b',max_length=512,batch_size=32):
        import torch
        from transformers import AutoModel,AutoTokenizer
        if not torch.cuda.is_available():raise RuntimeError('CUDA allocation required')
        if not (Path(path)/'checkpoint-provenance.json').exists():raise ValueError('Verify official checkpoint first')
        self.torch=torch;self.max_length=max_length;self.batch_size=batch_size
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False,padding_side='left')
        self.model=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()

    def encode(self,texts,query=False,*,mean_pool=False,group_size=None):
        torch=self.torch;rows=[]
        inputs=[('Instruct: '+TASK+'\nQuery:'+text) if query else text for text in texts]
        with torch.inference_mode():
            for start in range(0,len(inputs),self.batch_size):
                batch=self.tokenizer(inputs[start:start+self.batch_size],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt').to('cuda')
                if group_size is not None:
                    if group_size <= 0 or len(inputs) > self.batch_size or len(inputs) % group_size:
                        raise ValueError("Each embedding group must have the same size and fit in one forward")
                    # Recreate token positions from each group's independently padded input.
                    mask = batch["attention_mask"]
                    width = mask.shape[1]
                    group_widths = mask.sum(dim=1).reshape(-1, group_size).amax(dim=1)
                    extra_padding = (width - group_widths).repeat_interleave(group_size)
                    positions = torch.arange(width, device=mask.device) - extra_padding[:, None]
                    batch["position_ids"] = positions.clamp_min(0)
                hidden=self.model(**batch).last_hidden_state
                if mean_pool:
                    mask=batch['attention_mask'].unsqueeze(-1)
                    pooled=(hidden.float()*mask).sum(dim=1)/mask.sum(dim=1)
                else:
                    pooled=hidden[:,-1,:].float()
                vectors=torch.nn.functional.normalize(pooled,p=2,dim=1)
                array=vectors.cpu().numpy()
                if not np.isfinite(array).all():raise ValueError('Nonfinite embedding')
                rows.append(array)
        return np.concatenate(rows).astype(np.float32,copy=False)
