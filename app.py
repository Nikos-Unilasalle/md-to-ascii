import streamlit as st
import mistletoe
import re

# --- Dépendance pour la césure ---
try:
    import pyphen
except ImportError:
    st.error("La bibliothèque 'pyphen' est requise. Veuillez l'installer avec : pip install pyphen")
    st.stop()

# --- Pré-processeur (inchangé) ---
HR_PLACEHOLDER = "-HR-"
TITLE_MARKER = "@@TITLE@@"


def preprocess_markdown(text):
    text = re.sub(r'^(?!#)(.+)\n(-{3,})$', r'## \1', text, flags=re.MULTILINE)
    lines = text.split('\n')
    processed_lines = [HR_PLACEHOLDER if re.fullmatch(r'-{3,}', line.strip()) else line for line in lines]
    return "\n".join(processed_lines)


# --- Cœur de la logique de conversion (VOTRE CLASSE, INTÉGRÉE) ---

class AsciiRenderer:
    def __init__(self, justification='Non-justifié', line_width=70, use_hyphenation=True):
        self.justification = justification
        self.line_width = line_width
        self.use_hyphenation = use_hyphenation
        try:
            self.hyphenator = pyphen.Pyphen(lang='fr_FR')
        except Exception:
            # En cas d'échec (ex: dictionnaire manquant), on désactive la césure
            self.hyphenator = None
            self.use_hyphenation = False

    # ---------- Wrapping + Justification core ----------

    def _hyphen_positions(self, word):
        """Retourne les positions de césure possibles dans 'word' (pyphen)"""
        if not self.hyphenator:
            return []
        # La césure n'est pertinente que pour les mots assez longs
        if len(word) < 6:
            return []
        return list(self.hyphenator.positions(word))

    def _split_with_hyphen(self, word, max_len):
        """
        Tente de couper 'word' pour que la première partie + '-' tienne dans max_len.
        Retourne (part1_with_dash, part2) ou (None, None) si aucune coupe utile.
        """
        positions = self._hyphen_positions(word)
        for i in reversed(positions):
            # On ne coupe pas s'il reste moins de 3 caractères
            if i > 2 and len(word) - i > 2:
                part1 = word[:i] + "-"
                part2 = word[i:]
                if len(part1) <= max_len:
                    return part1, part2
        return None, None

    def _wrap_words(self, text, width, enable_hyphen):
        """
        Emballe 'text' en lignes <= width.
        Retourne une liste de lignes (chaînes). Ne fait PAS la justification ici.
        """
        words = text.split()
        lines = []
        current_line = ""

        while words:
            word = words.pop(0)
            candidate = word if not current_line else f"{current_line} {word}"

            if len(candidate) <= width:
                current_line = candidate
                continue

            # Le mot ne rentre pas. On doit traiter la ligne actuelle et le mot.
            if current_line:
                lines.append(current_line)

            # Maintenant, on traite le mot qui déborde sur une nouvelle ligne
            if len(word) > width and enable_hyphen:
                # Le mot seul est trop long, on tente de le couper
                head, tail = self._split_with_hyphen(word, width)
                if head:
                    lines.append(head)
                    words.insert(0, tail)  # On remet le reste au début de la liste
                    current_line = ""
                else:  # Pas de coupe possible, on met le mot et on continue
                    current_line = word
            else:
                # Le mot n'est pas trop long, il commence juste la nouvelle ligne
                current_line = word

        if current_line:
            lines.append(current_line)
        return lines

    def _justify_line(self, line, width):
        """
        Renvoie la version justifiée de 'line' à 'width'.
        Ne justifie pas si un seul mot, ou si la ligne finit par '-' (césure).
        """
        line = line.rstrip()
        if not line or line.endswith("-"):
            return line

        words = line.split()
        if len(words) <= 1:
            return line

        total_chars = sum(len(w) for w in words)
        spaces_needed = width - total_chars
        if spaces_needed <= 0:
            return line  # Déjà remplie ou trop pleine

        gaps = len(words) - 1
        base_spaces, extra_spaces = divmod(spaces_needed, gaps)

        justified_line = ""
        for i, w in enumerate(words[:-1]):
            num_spaces = base_spaces + (1 if i < extra_spaces else 0)
            justified_line += w + (" " * num_spaces)
        justified_line += words[-1]

        return justified_line

    def _format_block(self, text, width, indent="", justify=False, enable_hyphen=False):
        """
        Orchestre le wrapping et la justification pour un bloc de texte.
        """
        wrapped_lines = self._wrap_words(text, width, enable_hyphen)
        if not wrapped_lines:
            return ""

        output_lines = []
        for i, line in enumerate(wrapped_lines):
            # La dernière ligne n'est jamais justifiée
            if justify and i < len(wrapped_lines) - 1:
                formatted_line = self._justify_line(line, width)
            else:
                formatted_line = line

            # On ajoute l'indentation (qui peut être le préfixe de la ligne)
            # à la première ligne du bloc généré, et des espaces pour les suivantes.
            if i == 0:
                output_lines.append(indent + formatted_line)
            else:
                # L'indentation pour les lignes suivantes est constituée d'espaces
                # pour s'aligner avec le début du contenu de la première ligne.
                subsequent_indent = ' ' * len(indent)
                output_lines.append(subsequent_indent + formatted_line)

        return "\n".join(output_lines)

    # ---------- Mistletoe rendering ----------

    def render(self, node):
        if node is None: return ""
        node_class_name = node.__class__.__name__
        render_method = getattr(self, f"render_{node_class_name.lower()}", self.render_default)
        return render_method(node)

    def render_paragraph(self, node):
        raw_text = "".join(self.render(child) for child in node.children)
        if not raw_text.strip(): return ""

        output_paragraph = []
        for line in raw_text.split('\n'):
            if not line.strip():
                output_paragraph.append("")
                continue

            match = re.match(r'^(\s*([-*•]|\d+\.)\s*)', line)
            prefix = match.group(0) if match else ""
            content = line[len(prefix):]

            content_width = self.line_width - len(prefix)
            justify = (self.justification == 'Justifié')
            use_hyphen = self.use_hyphenation and justify

            formatted_block = self._format_block(
                content,
                width=content_width,
                indent=prefix,  # C'est le préfixe de la PREMIÈRE ligne
                justify=justify,
                enable_hyphen=use_hyphen
            )
            output_paragraph.append(formatted_block)

        # Le _format_block gère maintenant l'indentation, donc on join simplement
        return "\n".join(output_paragraph) + "\n\n"

    # --- Les autres méthodes render_* sont simplifiées ou utilisent la nouvelle logique ---

    def render_heading(self, node):
        if not node.children: return ""
        text = "".join(self.render(child) for child in node.children).strip().upper()
        if node.level == 1:
            title_block = f"+-{'-' * len(text)}-+\n| {text} |\n+-{'-' * len(text)}-+"
        elif node.level == 2:
            title_block = f"{text}\n{'=' * len(text)}"
        elif node.level == 3:
            title_block = f"{text}\n{'-' * len(text)}"
        else:
            title_block = f"{text}"
        return f"{TITLE_MARKER}{title_block}\n\n"

    def render_linebreak(self, node):
        return "\n"

    def render_rawtext(self, node):
        return node.content if hasattr(node, 'content') else ""

    def render_default(self, node):
        return "".join(self.render(child) for child in node.children) if hasattr(node,
                                                                                 'children') and node.children is not None else ""

    def render_list(self, node):
        output_list = []
        is_ordered = hasattr(node, 'start') and node.start is not None
        start_num = node.start if is_ordered else 1

        for i, item in enumerate(node.children, start=start_num):
            prefix = f"  {i}. " if is_ordered else "  - "
            content = "".join(self.render(child) for child in item.children).strip()
            content_width = self.line_width - len(prefix)
            justify = (self.justification == 'Justifié')
            use_hyphen = self.use_hyphenation and justify

            # On réutilise notre puissant _format_block
            formatted_item = self._format_block(
                content,
                width=content_width,
                indent=prefix,
                justify=justify,
                enable_hyphen=use_hyphen
            )
            output_list.append(formatted_item)
        return "\n".join(output_list) + "\n\n"

    def render_table(self, node):  # Inchangé
        def get_cell_content(cell):
            return "".join(self.render(child) for child in cell.children).strip()

        if not hasattr(node, 'header') or not node.header.children: return ""
        header_content = [get_cell_content(cell) for cell in node.header.children]
        body_rows_content = [[get_cell_content(cell) for cell in row.children] for row in node.children]
        column_widths = [len(h) for h in header_content]
        for row in body_rows_content:
            for i, cell in enumerate(row):
                if i < len(column_widths): column_widths[i] = max(column_widths[i], len(cell))

        def render_table_row(row_data):
            return "| " + " | ".join([c.ljust(column_widths[i]) for i, c in enumerate(row_data)]) + " |"

        separator = "|-" + "-|-".join(['-' * w for w in column_widths]) + "-|"
        output_lines = [render_table_row(header_content), separator,
                        *[render_table_row(row) for row in body_rows_content]]
        return "\n".join(output_lines) + "\n\n"


# --- Fonction principale (inchangée) ---
def markdown_to_ascii(text, justification='Non-justifié', line_width=70, use_hyphenation=True):
    if not text or not text.strip(): return ""
    preprocessed_text = preprocess_markdown(text)
    doc = mistletoe.Document(preprocessed_text)
    renderer = AsciiRenderer(justification, line_width, use_hyphenation)
    raw_output = "".join(renderer.render(child) for child in doc.children)
    output_with_markers = raw_output.replace(HR_PLACEHOLDER, '-' * line_width)
    spaced_output = re.sub(r'\s*(' + re.escape(TITLE_MARKER) + r')', r'\n\n\1', output_with_markers)
    final_output = spaced_output.replace(TITLE_MARKER, '')
    final_output = re.sub(r'\n{3,}', '\n\n', final_output)
    return final_output.strip()


# --- Interface utilisateur avec Streamlit (inchangée) ---
st.set_page_config(page_title="Markdown to ASCII", layout="wide")
st.title("📄 Convertisseur Markdown vers ASCII")
st.write("Collez votre texte Markdown pour le transformer en ASCII, avec des options de justification et de césure.")

default_md = """- Finalisation du projet: Amélioration de la structure du code, documentation complète et présentation aux utilisateurs. Note : 3.5/5 *Commentaire : Cette section est le cœur du mémoire et elle est globalement très bien exécutée, avec une phase exploratoire rigoureuse et une bonne compréhension des différentes techniques de clustering. Cependant, plusieurs points appellent à une critique plus sévère : - La déclaration que la qualité des données est "satisfaisante", après avoir mentionné des cas avec 30% de valeurs manquantes pour les IP et une précision très variable pour les coordonnées, est optimiste et manque de nuance. L'impact de ces lacunes sur les résultats des clusters aurait pu être plus explicitement discuté. - Le choix initial d'écarter la dimension temporelle pour la création des clusters, tout en la mentionnant comme une piste d'amélioration potentielle, révèle une opportunité manquée d'intégrer une composante riche dès le départ, ce qui aurait pu complexifier mais enrichir l'analyse des regroupements.*"""

with st.sidebar:
    st.header("⚙️ Options de formatage")
    justification_mode = st.radio(
        "Justification des paragraphes", ('Non-justifié', 'Justifié')
    )
    use_hyphenation = st.checkbox(
        "Activer la césure des mots", value=True, help="Permet de couper les mots pour une meilleure justification."
    )
    line_width = st.number_input(
        "Largeur de ligne (caractères)", min_value=40, max_value=120, value=80, step=5
    )

col1, col2 = st.columns(2)
with col1:
    st.header("Votre Markdown")
    markdown_input = st.text_area("Entrez votre texte ici", height=600, value=default_md)
with col2:
    st.header("Résultat ASCII")
    if markdown_input:
        ascii_output = markdown_to_ascii(markdown_input, justification_mode, line_width, use_hyphenation)
        st.code(ascii_output, language="")
        st.download_button(
            label="📥 Télécharger le fichier .txt", data=ascii_output.encode('utf-8'),
            file_name="ASCII.txt", mime="text/plain"
        )
    else:
        st.info("Le résultat s'affichera ici.")
