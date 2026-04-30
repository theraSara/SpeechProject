from typing import Dict, Any

def filter_opt(opt: Dict[str, Any], tag: str):
    ret: Dict[str, Any] = {}

    for k, v in opt.items():
        if not k.startswith(f'{tag}.'): continue
        _, cfg = k.split(f'{tag}.', maxsplit=1)
        ret[cfg] = v

    return ret

