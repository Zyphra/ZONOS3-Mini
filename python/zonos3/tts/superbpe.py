"""Frozen SuperBPE inference convention, identical to the training loader.

Phoneme-token checkpoints (lingua data.phoneme_tokens_path, text_extra_tokens rows after the text
padding row) add ``phonemes`` to the spec: a pronunciation group in the text, written
``open_marker`` + IPA + ``close_marker`` (U+E000 / U+E001), encodes as open_id, one id per IPA
code point of ``symbols`` (open_id + 2 + index) and close_id, the text between groups as
SuperBPE on its own -- lingua apps.zonos.text_tokenization._phoneme_payload_ids. The phoneme block
is the last block of the extra rows: right after the padding row (open_id = text_vocab + 1), or
after the speech-edit marker rows of an edit checkpoint (open_id = text_vocab + 1 + markers).
"""
import hashlib,json
from functools import lru_cache
from pathlib import Path
PHONEME_OPEN,PHONEME_CLOSE="\ue000","\ue001"  # lingua apps.zonos.phonemes markers (the spec carries its own)
@lru_cache(maxsize=8)
def tokenizer(path,sha,size):
    from tokenizers import Tokenizer
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha: raise ValueError("SuperBPE SHA mismatch")
    t=Tokenizer.from_str(raw.decode())
    if t.get_vocab_size(with_added_tokens=True)!=size or t.padding is not None or t.truncation is not None: raise ValueError("SuperBPE artifact mismatch")
    return t
def validate_spec(spec,text_vocab):
    if (spec['kind'],spec['offset'],spec['bos'],spec['eos'])!=('superbpe',4,2,3) or text_vocab!=spec['vocab_size']+4: raise ValueError("SuperBPE ID convention mismatch")
    tokenizer(spec['path'],spec['sha256'],spec['vocab_size'])
    phonemes=spec.get('phonemes')
    if phonemes is not None:
        symbols=phonemes['symbols']
        if phonemes['open_id']<text_vocab+1 or phonemes['close_id']!=phonemes['open_id']+1: raise ValueError("Phoneme token IDs must follow the text padding row (and any edit marker rows)")
        if any(len(s)!=1 for s in symbols) or len(set(symbols))!=len(symbols) or {phonemes['open_marker'],phonemes['close_marker']}&set(symbols): raise ValueError("Invalid phoneme symbols")
def phoneme_token_count(spec):
    """Text rows a spec's phoneme tokens take (0 without phonemes)."""
    phonemes=spec.get('phonemes') if spec else None
    return 0 if phonemes is None else 2+len(phonemes['symbols'])
def rows_before_phonemes(spec,text_vocab):
    """Extra text rows between the padding row and the phoneme block (edit marker rows)."""
    phonemes=spec.get('phonemes') if spec else None
    return 0 if phonemes is None else int(phonemes['open_id'])-int(text_vocab)-1
def _bpe(text,spec):
    return [i+4 for i in tokenizer(spec['path'],spec['sha256'],spec['vocab_size']).encode(text,add_special_tokens=False).ids]
def payload_ids(text,spec):
    phonemes=spec.get('phonemes')
    if phonemes is None:
        return _bpe(text,spec)
    open_,close=phonemes['open_marker'],phonemes['close_marker']
    if open_ not in text and close not in text:
        return _bpe(text,spec)
    phone_ids={s:phonemes['open_id']+2+i for i,s in enumerate(phonemes['symbols'])}
    ids,pos=[],0
    while pos<len(text):
        start=text.find(open_,pos)
        stretch=text[pos:] if start<0 else text[pos:start]
        if close in stretch: raise ValueError("Unbalanced phoneme group markers")
        if stretch: ids.extend(_bpe(stretch,spec))
        if start<0: break
        end=text.find(close,start+1)
        group=text[start+1:end] if end>=0 else ""
        if end<0 or not group or open_ in group: raise ValueError("Unbalanced or empty phoneme group")
        unknown=sorted(set(group)-phone_ids.keys())
        if unknown: raise ValueError(f"Phones outside the inventory: {unknown}")
        ids.append(phonemes['open_id']); ids.extend(phone_ids[s] for s in group); ids.append(phonemes['close_id'])
        pos=end+1
    return ids
def text_ids(text,spec):
    return [2]+payload_ids(text,spec)+[3]
def load_spec(model_path):
    from zonos3.utils.release import resolve_dir_or_local,resolve_tokenizer_path  # RELEASE_HF_20261021
    model_dir=resolve_dir_or_local(str(model_path))
    model=json.loads((Path(model_dir)/'params.json').read_text())['model']
    spec=model.get('text_tokenizer_spec')
    if spec is not None:
        spec=dict(spec,path=resolve_tokenizer_path(spec['path'],model_dir)); validate_spec(spec,model['text_vocab'])
        if spec.get('phonemes') is not None and rows_before_phonemes(spec,model['text_vocab'])+phoneme_token_count(spec)!=int(model.get('text_extra_tokens',0)): raise ValueError("Phoneme tokens do not match text_extra_tokens (they must be the last extra rows)")
    return spec
