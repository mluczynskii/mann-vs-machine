import re

def split_pascal(x):
    """Inserts spaces in between words contained in a phrase
    written in Pascal case:
    
    'PascalCase' -> 'Pascal Case'."""

    words = []
    beginning = 0
    for i, c in enumerate(x):
        if i > 0 and str.isupper(c):
            words.append(x[beginning:i])
            beginning = i
    words.append(x[beginning:])

    return " ".join(words)


expr = r"\b([A-Z][a-z]*)*([A-Z][a-z]*)([A-Z][a-z]+)+([A-Z][a-z]*)*\b"
pattern = re.compile(expr)

def split_pascals(x):
    """Transforms x by performing split_pascal on all substrings
      (delimited by word boundaries)
      of x that seem to be written in Pascal case."""
    
    res = []
    last = 0
    for m in re.finditer(pattern, x):
        start, end = m.start(), m.end()

        # Leave hashtags as they are
        if not (start > 0 and x[start - 1] == "#"):
            res.append(x[last:start])
            res.append(split_pascal(x[start:end]))
            last = end
            
    res.append(x[last:])
    
    return "".join(res)


def tokenize(x, returntype="bag", splitpascals=False):
    """A custom tokenization function. 
    
    Tokenization consists of:
    - splitting phrases written in Pascal case (optional, if splitpascals
    is True) (note: this does not transform hashtags!),
    - filtering out nonalphabetic characters,
    - converting all characters to uniform case.

    If returntype is 'set', returns a set of strings.
    If returntype is 'bag', returns a list of strings.
    Other values of this parameter are not supported.
    """

    if splitpascals:
        x = split_pascals(x)
    
    toks = str.casefold("".join(
        filter(lambda s: str.isalpha(s) or str.isspace(s),
                x))).split()
    
    if returntype == "bag":
        return toks
    elif returntype == "set":
        return set(toks)
    else:
        return None
