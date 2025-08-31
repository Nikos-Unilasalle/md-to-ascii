import streamlit as st
import mistletoe
import re

# --- Dépendance césure ---
try:
    import pyphen
except ImportError:
    st.error("Installez pyphen : pip install pyphen")
    st.stop()

# --- Constantes ---
HR_PLACEHOLDER = "-HR-"
TITLE_MARKER = "@@TITLE@@"
MATH_BLOCK_MARKER = "@@MATH_BLOCK_{}@@"
INLINE_MATH_MARKER = "@@INLINE_MATH_{}@@"

# --- LaTeX -> ASCII ---
SYMBOL_MAP = {
    '\\cdot': '*', '\\times': '*', '\\pm': '+/-', '\\le': '<=', '\\ge': '>=', '\\ne': '!=',
    '\\approx': '~=', '\\equiv': '===', '\\in': 'in', '\\notin': 'not in', '\\infty': 'infinity',
    '\\bmod': ' mod ', '\\circ': ' o ', '\\mapsto': '->', '\\forall': 'for all',
    '\\leftrightarrow': '<->', '\\to': '->', '\\rightarrow': '->', '\\Rightarrow': '=>',
    '\\gets': '<-', '\\leftarrow': '<-', '\\Leftarrow': '<=',
    '\\prime': "'", '\\ldots': '...', '\\cdots': '...', '\\star': '*', '\\hbar': 'h-bar',
    '\\|': '|', '||': '|', '\\quad': '    ', '\\qquad': '        ',
    '\\,': ' ', '\\;': ' ', '\\ ': ' ', '\\!': '',
    '\\lceil': 'ceil(', '\\rceil': ')', '\\lfloor': 'floor(', '\\rfloor': ')',
    '\\langle': '<', '\\rangle': '>',
    '\\alpha': 'alpha', '\\beta': 'beta', '\\gamma': 'gamma', '\\Gamma': 'Gamma',
    '\\delta': 'delta', '\\Delta': 'Delta', '\\epsilon': 'epsilon', '\\zeta': 'zeta',
    '\\eta': 'eta', '\\theta': 'theta', '\\Theta': 'Theta', '\\iota': 'iota',
    '\\kappa': 'kappa', '\\lambda': 'lambda', '\\Lambda': 'Lambda', '\\mu': 'mu',
    '\\nu': 'nu', '\\xi': 'xi', '\\Xi': 'Xi', '\\pi': 'pi', '\\Pi': 'Pi', '\\rho': 'rho',
    '\\sigma': 'sigma', '\\Sigma': 'Sigma', '\\tau': 'tau', '\\upsilon': 'upsilon',
    '\\phi': 'phi', '\\Phi': 'Phi', '\\chi': 'chi', '\\psi': 'psi', '\\Psi': 'Psi',
    '\\omega': 'omega', '\\Omega': 'Omega', '\\ell': 'l', '\\∎': 'Q.E.D.'
}
SYMBOL_REGEX = re.compile('|'.join(re.escape(k) for k in SYMBOL_MAP.keys()))

def render_inline_math_to_ascii(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\\(boldsymbol|mathcal|mathbb|mathrm)\{([^}]+)\}', r'\2', s)
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = SYMBOL_REGEX.sub(lambda m: SYMBOL_MAP[m.group(0)], s)
    s = s.replace('\\(', '(').replace('\\)', ')').replace('{', '').replace('}', '')
    return re.sub(r'\s+', ' ', s).strip()

def _parse_latex_array(content: str) -> str:
    rows = content.split('\\\\')
    table = []
    for r in rows:
        r = r.strip()
        if not r: continue
        if '\\hline' in r:
            parts = [p.strip() for p in r.split('\\hline')]
            for p in parts:
                if not p: table.append('---H---')
                else: table.append([render_inline_math_to_ascii(c) for c in p.split('&')])
        else:
            table.append([render_inline_math_to_ascii(c) for c in r.split('&')])
    if not table: return ""
    n = max(len(r) for r in table if isinstance(r, list))
    widths = [0]*n
    for r in table:
        if isinstance(r, list):
            for i,c in enumerate(r):
                widths[i] = max(widths[i], len(c))
    out = []
    for r in table:
        if r == '---H---':
            out.append("+-" + "-+-".join('-'*w for w in widths) + "-+")
        else:
            r += [""]*(n-len(r))
            out.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(n)) + " |")
    return "\n".join(out)

def render_math_block_to_ascii(s: str) -> str:
    s = s.strip()
    if re.search(r'\\begin\{array\}.*?\\end\{array\}', s, re.DOTALL):
        m = re.search(r'\\begin\{array\}.*?\}(.*?)\\end\{array\}', s, re.DOTALL)
        return _parse_latex_array(m.group(1))
    if re.search(r'\\begin\{cases\}.*?\\end\{cases\}', s, re.DOTALL):
        m = re.search(r'\\begin\{cases\}(.*?)\\end\{cases\}', s, re.DOTALL)
        lines = m.group(1).strip().split('\\\\')
        return "\n".join("  "+render_inline_math_to_ascii(t) for t in lines if t.strip())
    return "\n".join("  "+render_inline_math_to_ascii(l) for l in s.splitlines() if l.strip())

# --- Prétraitement Markdown ---
def preprocess_markdown(text: str):
    mm = {}
    def repl_block(m):
        ph = MATH_BLOCK_MARKER.format(len(mm))
        mm[ph] = render_math_block_to_ascii(m.group(1))
        return f"\n\n{ph}\n\n"
    text = re.sub(r'\$\$(.*?)\$\$', repl_block, text, flags=re.DOTALL)

    def repl_inline(m):
        ph = INLINE_MATH_MARKER.format(len(mm))
        mm[ph] = render_inline_math_to_ascii(m.group(1))
        return ph
    text = re.sub(r'(?<![\$\\])\$([^$\n]+?)\$(?!\$)', repl_inline, text)

    text = re.sub(r'^(?!#)(.+)\n(-{3,})$', r'## \1', text, flags=re.MULTILINE)

    lines = text.split('\n')
    lines = [HR_PLACEHOLDER if re.fullmatch(r'-{3,}', l.strip()) else l for l in lines]
    return "\n".join(lines), mm

# --- Utilitaires ---
def indent_block(s: str, n: int) -> str:
    pad = " " * n
    return "\n".join((pad + l if l else l) for l in s.splitlines())

def _strip_parent_prefix_leaks(text: str, parent_prefix: str) -> str:
    """
    Si une ligne (sauf la 1re) commence par le VRAI préfixe parent ('  1. '),
    on le remplace par des espaces de même largeur. On NE touche PAS au '2. ' de la sous-liste.
    """
    lines = text.splitlines()
    if not lines: return text
    pad = " " * len(parent_prefix)
    fixed = [lines[0]]
    for ln in lines[1:]:
        if ln.startswith(parent_prefix):
            fixed.append(pad + ln[len(parent_prefix):])
        else:
            fixed.append(ln)
    return "\n".join(fixed)

# --- Rendu ASCII ---
class AsciiRenderer:
    def __init__(self, line_width=70, use_hyphenation=True):
        self.line_width = line_width
        self.use_hyphenation = use_hyphenation
        try:
            self.hyphenator = pyphen.Pyphen(lang='fr_FR')
        except Exception:
            self.hyphenator, self.use_hyphenation = None, False

    def _hyphen_positions(self, w):
        if not self.hyphenator or len(w) < 6 or w.startswith("@@"): return []
        return list(self.hyphenator.positions(w))

    def _split_with_hyphen(self, w, max_len):
        for i in reversed(self._hyphen_positions(w)):
            if i > 2 and len(w) - i > 2:
                p1, p2 = w[:i] + "-", w[i:]
                if len(p1) <= max_len: return p1, p2
        return None, None

    def _wrap_words(self, text, width):
        words, lines, cur = text.split(), [], ""
        while words:
            w = words.pop(0)
            cand = w if not cur else f"{cur} {w}"
            if len(cand) <= width:
                cur = cand; continue
            if cur: lines.append(cur)
            while len(w) > width:
                head, tail = (None, None)
                if self.use_hyphenation: head, tail = self._split_with_hyphen(w, width)
                if head: lines.append(head); w = tail
                else:    lines.append(w[:width]); w = w[width:]
            cur = w
        if cur: lines.append(cur)
        return lines

    def _format_block(self, text, indent=""):
        segs = text.split("\n"); cw = max(10, self.line_width - len(indent)); out = []
        for s in segs:
            fused = re.sub(r'[ \t]+', ' ', s).strip()
            if fused == "": out.append(""); continue
            wrapped = self._wrap_words(fused, cw)
            for i, w in enumerate(wrapped):
                out.append((indent if i == 0 else ' '*len(indent)) + w)
        return "\n".join(out)

    def render(self, node):
        fn = getattr(self, f"render_{node.__class__.__name__.lower()}", self.render_default)
        return fn(node)

    def render_rawtext(self, node): return getattr(node, 'content', "")
    def render_linebreak(self, node): return "\n"
    def render_default(self, node): return "".join(self.render(c) for c in getattr(node, 'children', []) or [])
    def render_strong(self, node): return f"**{''.join(self.render(c) for c in node.children)}**"
    def render_emphasis(self, node): return f"*{''.join(self.render(c) for c in node.children)}*"

    def render_paragraph(self, node):
        raw = "".join(self.render(c) for c in node.children)
        if re.fullmatch(MATH_BLOCK_MARKER.format(r'\d+'), raw.strip()): return raw.strip() + "\n\n"
        if not raw.strip(): return ""
        return self._format_block(raw) + "\n\n"

    def render_list(self, node):
        out = []
        is_ordered = hasattr(node, 'start') and node.start is not None
        start_num = node.start if is_ordered else 1

        for idx, item in enumerate(node.children, start=start_num):
            parent_prefix = f"  {idx}. " if is_ordered else "  - "
            blocks, first_text_done = [], False

            for child in item.children:
                kind = child.__class__.__name__
                if kind == "Paragraph":
                    inline = "".join(self.render(gc) for gc in child.children)
                    indent = parent_prefix if not first_text_done else " " * len(parent_prefix)
                    blocks.append(self._format_block(inline, indent=indent))
                    first_text_done = True
                elif kind == "List":
                    sub = self.render(child).rstrip("\n")
                    blocks.append(indent_block(sub, len(parent_prefix)))
                else:
                    rendered = self.render(child).rstrip("\n")
                    if rendered:
                        blocks.append(indent_block(rendered, len(parent_prefix)))

            if not blocks: blocks = [parent_prefix.rstrip()]
            item_text = "\n".join(blocks)
            # >>> Empêche '1. 2.' : enlève le préfixe du parent sur lignes suivantes
            item_text = _strip_parent_prefix_leaks(item_text, parent_prefix)
            out.append(item_text)

        return "\n".join(out) + "\n\n"

    def render_heading(self, node):
        text = "".join(self.render(c) for c in node.children).strip().upper()
        if not text: return ""
        width = max(20, self.line_width)
        if node.level == 1:
            inner = max(1, width - 4)
            wrapped = self._wrap_words(text, inner)
            top = "+-" + "-"*(width-4) + "-+"
            lines = [top]
            for w in wrapped:
                pad = inner - len(w); l = pad // 2; r = pad - l
                lines.append(f"| {' '*l}{w}{' '*r} |")
            lines.append(top)
            block = "\n".join(lines)
        else:
            underline = "="*width if node.level == 2 else "-"*width
            wrapped = self._wrap_words(text, width)
            block = "\n".join(wrapped) + "\n" + underline
        return f"{TITLE_MARKER}{block}\n\n"

    def render_table(self, node):
        def cell(c): return "".join(self.render(ch) for ch in c.children).replace('\\hline','').strip()
        if not getattr(node, 'header', None) or not node.header.children: return ""
        header = [cell(c) for c in node.header.children]
        body = [[cell(c) for c in r.children] for r in node.children]
        colw = [len(h) for h in header]
        for r in body:
            for i,c in enumerate(r):
                if i < len(colw): colw[i] = max(colw[i], len(c))
        def rowline(data): return "| " + " | ".join(data[i].ljust(colw[i]) for i in range(len(colw))) + " |"
        sep = "|-" + "-|-".join('-'*w for w in colw) + "-|"
        return "\n".join([rowline(header), sep, *[rowline(r) for r in body]]) + "\n\n"

# --- Justification ---
def justify_line(line: str, width: int) -> str:
    words = line.split()
    if len(words) <= 1: return line
    total = sum(len(w) for w in words)
    gaps = len(words) - 1
    need = width - total
    if need <= 0: return line
    base, extra = divmod(need, gaps)
    parts = []
    for i, w in enumerate(words[:-1]):
        parts.append(w + " " * (base + (1 if i < extra else 0)))
    parts.append(words[-1])
    return "".join(parts)

def post_process_and_justify(text: str, width: int, mode: str) -> str:
    if mode != 'Justifié': return text
    final = []
    blocks = re.split(r'(\n{2,})', text)
    for i in range(0, len(blocks), 2):
        block = blocks[i]; sep = blocks[i+1] if i+1 < len(blocks) else ""
        if not block.strip(): final.append(block + sep); continue
        lines = block.split('\n')
        first = lines[0] if lines else ""

        def looks_like_list(l0):
            s0 = l0.lstrip(' ')
            return s0.startswith(('- ', '* ', '+ ')) or re.match(r'\d+\.\s', s0)

        is_box = first.startswith("+-") and first.endswith("-+")
        is_table = first.startswith(('|', '+-')) or (first.startswith('  ') and ('| ' in first or '+-' in first))
        is_heading_text = bool(re.search(r'\n[=-]{10,}$', block, flags=re.M))

        if looks_like_list(first) or is_table or is_box or is_heading_text:
            final.append("\n".join(lines) + sep)
            continue

        if len(lines) <= 1:
            final.append(block + sep)
            continue

        out = []
        for j, line in enumerate(lines):
            if not line.strip(): out.append(""); continue
            indent = len(line) - len(line.lstrip(' '))
            eff = max(10, width - indent)
            last = (j == len(lines) - 1)
            too_short = len(line.strip()) < int(0.6 * eff)
            if last or too_short:
                out.append(line)
            else:
                out.append(' '*indent + justify_line(line.lstrip(' '), eff))
        final.append("\n".join(out) + sep)
    return "".join(final)

# --- API principale ---
def markdown_to_ascii(text, justification='Non-justifié', line_width=70, use_hyphenation=True):
    if not text or not text.strip(): return ""
    preprocessed, math_map = preprocess_markdown(text)
    doc = mistletoe.Document(preprocessed)
    renderer = AsciiRenderer(line_width, use_hyphenation)
    rendered = renderer.render(doc)
    justified = post_process_and_justify(rendered, line_width, justification)

    def back(m): return math_map.get(m.group(0), m.group(0))
    out = re.sub(r'@@(?:INLINE_MATH|MATH_BLOCK)_\d+@@', back, justified)
    out = out.replace(HR_PLACEHOLDER, '-' * line_width)
    out = re.sub(r'\s*(' + re.escape(TITLE_MARKER) + r')', r'\n\n\1', out)
    out = out.replace(TITLE_MARKER, '')
    out = re.sub(r'\n{3,}', '\n\n', out)
    return out.strip()

# --- UI ---
st.set_page_config(page_title="Markdown to ASCII", layout="wide")
st.title("📄 Convertisseur Markdown vers ASCII")

default_md = """### Titre de section

1. Une liste officielle.
   2. Un sous-item dont le contenu est lui aussi très long pour vérifier que le formatage s'applique correctement à l'intérieur des listes reconnues par le parseur.
   3. Encore un sous-item.
2. et voilà

- test 34
- test 2
- test
- test avec indentation
  -test sans espace

Ceci est un paragraphe normal.
Il continue ici, et cette ligne est volontairement très très longue pour s'assurer que le retour à la ligne automatique fonctionne correctement sans fusionner cette ligne avec la précédente ou la suivante.

La justification ne doit se faire que sur les lignes qui dépassent la limite de caractère
"""

with st.sidebar:
    st.header("⚙️ Options")
    justification_mode = st.radio("Justification des paragraphes", ('Non-justifié', 'Justifié'))
    use_hyphenation = st.checkbox("Activer la césure des mots", value=True)
    line_width = st.number_input("Largeur de ligne", min_value=40, max_value=120, value=80, step=5)

col1, col2 = st.columns(2)
with col1:
    st.header("Votre Markdown")
    md_in = st.text_area("Texte", height=600, value=default_md)
with col2:
    st.header("Résultat ASCII")
    if md_in:
        ascii_out = markdown_to_ascii(md_in, justification_mode, line_width, use_hyphenation)
        st.code(ascii_out, language="")
        st.download_button("📥 Télécharger .txt", ascii_out.encode("utf-8"), "ASCII.txt", "text/plain")
    else:
        st.info("Le résultat s'affichera ici.")
