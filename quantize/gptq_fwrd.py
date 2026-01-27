import torch, torch.nn as nn, torch.nn.functional as F
import logging
import gc
from quantize.E2M_ATQ import Ternarization
from quantize.tra_gptq import TRAGPTQ

def find_qlayers(module, layers=[torch.nn.Linear], name=''):
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@torch.no_grad()
def twla_fwrd(model, tokenizer, dataloader, dev, args):
    print('-----E2M-ATQ Quantization-----')
    if args.group_size is None:
        args.group_size = -1
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = model.lm_head.weight.dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "inps": inps, "kwargs": None}

    class Catcher(nn.Module):
        """Helper class to capture input data and kwargs."""
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["kwargs"] = kwargs
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError
        
        def __getattr__(self, name):
            """Forward attribute access to the wrapped module."""
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
            
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            try:
                model(batch[0].to(dev))
            except:
                texts = batch["text"]
                queries = [query for query in texts]
                inputs = tokenizer(queries, return_tensors="pt", truncation=True, max_length=model.seqlen, padding=True).to('cuda')
                inputs['input_ids'] =  F.pad(inputs['input_ids'], (0, 2048 - inputs['input_ids'].shape[1]))
                model(inputs['input_ids'])
        except:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    if attention_mask is not None:
        attention_mask = attention_mask[:1]
    #position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
        ['self_attn.q_proj','self_attn.k_proj', 'self_attn.v_proj'],
        ['self_attn.o_proj'],
        ['mlp.up_proj', 'mlp.gate_proj'],
        ['mlp.down_proj']
    ]

    layer_kwargs = cache["kwargs"] or {}
    # Ensure tensors in kwargs are on correct device
    for k, v in list(layer_kwargs.items()):
        if isinstance(v, torch.Tensor):
            layer_kwargs[k] = v.to(dev)
    print ('layers length: ', len(layers))
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.wbits
                layer_weight_sym = args.symmetric
                if 'lm_head' in name:
                    print ('attention')
                    layer_weight_bits = 16
                    continue
                group_or_block = 128
                braq_quantizer = Ternarization(
                    subset[name].weight,
                    groupsize=group_or_block,
                )
                gptq[name] = TRAGPTQ(
                    layer=subset[name],
                    braq_quantizer=braq_quantizer,
                    salient_metric=getattr(args, "salient_metric", "hessian"),  
                    disable_gptq=getattr(args, "disable_gptq_core", False),   
                    order2_group=getattr(args, "order2_group", True),
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name))) # 搜集数据
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in gptq:
                logging.info(f'{i} {name}')
                logging.info("Quantizing ...")
                info = gptq[name].fasterquant(
                    blocksize=getattr(args, "blocksize", 128),
                    percdamp=getattr(args, "percdamp", 0.1),
                    orders=(1, 1, 2),
                    num_p=getattr(args, "num_p", 1),
                )
                gptq[name].free()
        print ('i: ', i)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer.cpu()
        
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    
    if True: 
        print(f"\nhead:", flush=True, end=" ")
        model.lm_head = model.lm_head.to(device='cuda', dtype=torch.float32)
        torch.cuda.empty_cache()
        print ('head finished')

    model.config.use_cache = use_cache
    clear_cache()
    print('-----E2M-ATQ Quantization Done-----\n')
    return quantizers
