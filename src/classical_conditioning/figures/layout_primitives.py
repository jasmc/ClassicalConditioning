"""Pure layout addressing and deterministic serialization shared by commands."""
import json

def encode(value): return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()

def find(root, identity):
    matches=[n for n in root.iter() if n.get('id')==identity]
    if len(matches)!=1: raise ValueError(f'Expected one element: {identity}')
    return matches[0]
