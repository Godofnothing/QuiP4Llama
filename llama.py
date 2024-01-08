import os
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import wandb
from transformers import AutoModelForCausalLM

from gptq import *
from bal import Balance
from near import Nearest
from modelutils import *
from quant import *

from tqdm import tqdm


@contextmanager
def suspend_nn_inits():
    skip = lambda *args, **kwargs: None
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def get_llama(model):
    model = AutoModelForCausalLM.from_pretrained(model, low_cpu_mem_usage=True, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev, args):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    quantizers = {}
    errors, Hmags, times = [], [], []
    for i in tqdm(range(len(layers)), desc="Quantizing layer"):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        quant_method = {}
        # Initialize Quant Method and Compute H
        for name in subset:
            if args.quant == "gptq":
                quant_method[name] = GPTQ(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits, perchannel=True, sym=False, qfn=args.qfn, mse=False)
            elif args.quant == "nearest":
                quant_method[name] = Nearest(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits, perchannel=True, sym=False, qfn=args.qfn, mse=False)
            elif args.quant in ["allbal", "ldlq", "ldlqRG", "ldlbal_admm"]:
                quant_method[name] = Balance(subset[name])
                quant_method[name].configure(args.quant, args.wbits, args.npasses, unbiased=args.unbiased)
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits, perchannel=True, sym=False, qfn=args.qfn, mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
        for name in subset:
            quant_method[name].post_batch()

        # Quantize Weights
        for name in subset:
            quant_method[name].preproc(
                preproc_gptqH=args.pre_gptqH,
                percdamp=args.percdamp,
                preproc_rescale=args.pre_rescale,
                preproc_proj=args.pre_proj,
                preproc_proj_extra=args.pre_proj_extra,
            )
            if args.quant == "gptq":
                quant_method[name].fasterquant(groupsize=args.groupsize)
            elif args.quant in ["allbal", "ldlq", "ldlqRG", "ldlbal_admm"]:
                quant_method[name].fasterquant(lazy_batch=args.lazy_batch)
            elif args.quant == "nearest":
                quant_method[name].fasterquant()
            quantizers["model.layers.%d.%s" % (i, name)] = quant_method[name].quantizer

            errors.append(quant_method[name].error)
            times.append(quant_method[name].time)
            Hmags.append(quant_method[name].Hmag)
            quant_method[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del quant_method
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    print(f"Total quant time: {sum(times):.2f}s")

    return quantizers, errors


@torch.no_grad()
def llama_eval(odel, testenc, dev, dataset_name):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in tqdm(range(len(layers))):
        # print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\n{dataset_name} perplexity = {ppl.item():.4f}\n")

    if args.wandb:
        wandb.log({dataset_name: ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="model to load")
    parser.add_argument(
        "dataset", type=str, default=None, help="Where to extract calibration data from."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument(
        "--percdamp", type=float, default=0.01, help="Percent of the average Hessian diagonal to use for dampening."
    )
    parser.add_argument(
        "--quant",
        choices=["allbal", "ldlq", "ldlqRG", "ldlbal_admm", "nearest", "gptq"],
        default="nearest",
        help="Which quantization method to use.",
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument("--npasses", type=int, default=0, help="number passes to repeat balance loop over 1-d.")
    parser.add_argument(
        "--groupsize", type=int, default=-1, help="Groupsize to use for quantization; default uses full row."
    )
    parser.add_argument("--pre_gptqH", action="store_true", help="preprocessing")
    parser.add_argument("--pre_rescale", action="store_true", help="preprocessing")
    parser.add_argument("--pre_proj", action="store_true", help="preprocessing")
    parser.add_argument(
        "--pre_proj_extra", type=int, default=0, choices=[0, 1, 2], help="Extra options to control pre_proj step."
    )
    parser.add_argument("--qfn", type=str, default="a", help="qfn: a is default, b is sym incoherent based")
    parser.add_argument("--save", type=str, default="", help="Save quantized checkpoint under this name.")
    parser.add_argument(
        "--check", action="store_true", help="Whether to compute perplexity during benchmarking for verification."
    )
    parser.add_argument("--proxy_only", action="store_true", help="Only compute proxy objective (w^T H w)")
    parser.add_argument("--unbiased", action="store_true", help="unbiased")
    parser.add_argument("--incoh_processing", action="store_true", help="incoherence processing")
    parser.add_argument("--lazy_batch", action="store_true", help="lazy batch updates in blocks as used in OPTQ")
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb.")

    args = parser.parse_args()
    # defaults to incoherence processing
    if args.incoh_processing:
        args.pre_gptqH = True
        args.pre_rescale = True
        args.pre_proj = True
        args.proj_extra = 1
        args.qfn = "b"

    if args.wandb:
        wandb.init(
            name=os.environ.get("WANDB_NAME", "QuIP_run"),
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )
        wandb.run.log_code(".")

    model = get_llama(args.model)
    model.eval()

    dataloader = get_loaders(
        args.dataset, 
        nsamples=args.nsamples, 
        seed=args.seed, 
        seqlen=model.seqlen,
        model_path=args.model
    )

    if args.wbits < 16:
        # Preprocessing flags
        if args.qfn == "b":
            assert args.pre_proj is True
        print(
            f"Preprocessing flags: gptqH:{args.pre_gptqH}, rescale:{args.pre_rescale}, proj:{args.pre_proj}, proj_extra:{args.pre_proj_extra}, qfn:{args.qfn}"
        )
        print(f"using lazy_batch updates: {args.lazy_batch}")
        # LDL checks
        if ("ldl" in args.quant) and args.unbiased and (args.npasses > 0):
            print(f"LDL NOTE: unbiased + {args.npasses} npasses. NOT TRULY UNBIASED.")

        tick = time.time()
        quantizers, errors = llama_sequential(model, dataloader, DEV, args)
        print(f"Total quant + H time elapsed: {time.time() - tick:.2f}s")
        print("")
        print(
            f"Proxy Summary: Qmethod:{args.quant}, Unbiased: {args.unbiased}, W:{args.wbits}, NPass:{args.npasses}"
        )
        print("Quantization done.")
        print("")

    if args.save:
        torch.save(model.state_dict(), args.save)

    if not args.proxy_only:
        for dataset in ["wikitext2", "ptb-new", "c4-new"]:
            testloader = get_loaders(
                dataset, 
                seed=args.seed, 
                seqlen=model.seqlen,
                eval_mode=True,
                model_path=args.model
            )
            print(dataset)
            llama_eval(model, testloader, DEV, dataset)
