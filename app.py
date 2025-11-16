import streamlit as st
import mistletoe
import re

try:
    import pyphen
except ImportError:
    st.error("La bibliothèque 'pyphen' est requise. Veuillez l'installer avec : pip install pyphen")
    st.stop()

HR_PLACEHOLDER = "-HR-"
TITLE_MARKER = "@@TITLE@@"


def preprocess_markdown(text):
    text = re.sub(r'^(?!#)(.+)\n(-{3,})$', r'## \1', text, flags=re.MULTILINE)
    lines = text.split('\n')
    processed_lines = [HR_PLACEHOLDER if re.fullmatch(r'-{3,}', line.strip()) else line for line in lines]
    return "\n".join(processed_lines)


class AsciiRenderer:
    def __init__(self, justification='Non-justifié', line_width=70, use_hyphenation=True):
        self.justification = justification
        self.line_width = line_width
        self.use_hyphenation = use_hyphenation
        try:
            self.hyphenator = pyphen.Pyphen(lang='fr_FR')
        except Exception:
            self.hyphenator = None
            self.use_hyphenation = False

    # ---------- Wrapping + Justification core ----------

    def _hyphen_positions(self, word):
        """Retourne les positions de césure possibles dans 'word' (pyphen)"""
        if not self.hyphenator:
            return []
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

            if current_line:
                lines.append(current_line)

            if len(word) > width and enable_hyphen:
                head, tail = self._split_with_hyphen(word, width)
                if head:
                    lines.append(head)
                    words.insert(0, tail)
                    current_line = ""
                else:
                    current_line = word
            else:
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

            if i == 0:
                output_lines.append(indent + formatted_line)
            else:
                subsequent_indent = ' ' * len(indent)
                output_lines.append(subsequent_indent + formatted_line)

        return "\n".join(output_lines)

    def _distribute_column_widths_exact(self, max_widths, target_width):
        """
        Distribution proportionnelle qui GARANTIT que la somme = target_width.
        Respecte une largeur minimale de 3 caractères par colonne.
        """
        n_cols = len(max_widths)
        if n_cols == 0:
            return []

        MIN_COL_WIDTH = 3

        # Vérification du cas impossible
        if target_width < MIN_COL_WIDTH * n_cols:
            return [MIN_COL_WIDTH] * n_cols

        # Si toutes les colonnes sont vides
        total_max = sum(max_widths)
        if total_max == 0:
            base = target_width // n_cols
            widths = [base] * n_cols
            remainder = target_width - sum(widths)
            for i in range(remainder):
                widths[i] += 1
            return widths

        # Distribution proportionnelle initiale
        widths = []
        for w in max_widths:
            # Calcul proportionnel avec arrondi
            width = max(MIN_COL_WIDTH, int(round(w * target_width / total_max)))
            widths.append(width)

        # Ajuster l'erreur d'arrondi
        diff = target_width - sum(widths)

        if diff > 0:
            # Il manque de la largeur: l'ajouter aux colonnes les plus "compressées"
            # On calcule un ratio largeur/naturel pour voir qui a perdu le plus
            ratios = [(widths[i] / max_widths[i] if max_widths[i] > 0 else 1, i) for i in range(n_cols)]
            ratios.sort()  # Plus petit ratio = plus compressé
            for i in range(diff):
                idx = ratios[i % n_cols][1]
                widths[idx] += 1
        elif diff < 0:
            # Trop de largeur: la retirer des colonnes les plus larges
            sorted_idx = sorted(range(n_cols), key=lambda i: widths[i], reverse=True)
            for i in range(-diff):
                if widths[sorted_idx[i]] > MIN_COL_WIDTH:
                    widths[sorted_idx[i]] -= 1

        # Vérification finale (sécurité)
        final_sum = sum(widths)
        if final_sum != target_width:
            widths[-1] += target_width - final_sum

        return widths

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
                indent=prefix,
                justify=justify,
                enable_hyphen=use_hyphen
            )
            output_paragraph.append(formatted_block)

        return "\n".join(output_paragraph) + "\n\n"

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

    def render_inlinecode(self, node):
        """Gère le texte entre backticks (code inline)"""
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

            formatted_item = self._format_block(
                content,
                width=content_width,
                indent=prefix,
                justify=justify,
                enable_hyphen=use_hyphen
            )
            output_list.append(formatted_item)
        return "\n".join(output_list) + "\n\n"

    def render_table(self, node):
        def get_cell_content(cell):
            """Récupère le contenu texte d'une cellule, gère les backticks et autres formats"""
            if not cell or not hasattr(cell, 'children') or not cell.children:
                return ""

            parts = []
            for child in cell.children:
                parts.append(self.render(child))

            return "".join(parts).strip()

        # 1. Extraction des données du tableau
        if not hasattr(node, 'header') or not node.header.children:
            return ""

        header_cells = node.header.children
        n_cols = len(header_cells)
        if n_cols == 0:
            return ""

        header_content = [get_cell_content(cell) for cell in header_cells]
        body_rows_content = []

        for row in node.children:
            if hasattr(row, 'children') and row.children:
                # Limiter au nombre de colonnes du header
                cells = row.children[:n_cols]
                body_rows_content.append([get_cell_content(cell) for cell in cells])

        # 2. Calculer la largeur structurelle exacte
        # Format: | cell1 | cell2 | cell3 |
        # Largeur = sum(widths) + 3*n_cols (bordures et séparateurs)
        structural_width = 3 * n_cols

        # 3. Largeur disponible pour le contenu
        available_content_width = self.line_width - structural_width

        if available_content_width < 0:
            return f"[Erreur: line_width doit être >= {structural_width}]\n\n"

        # 4. Largeurs naturelles maximales
        max_widths = [len(h) for h in header_content]
        for row in body_rows_content:
            for i, cell in enumerate(row):
                if i < len(max_widths):
                    max_widths[i] = max(max_widths[i], len(cell))

        # 5. Calculer les largeurs finales des colonnes
        total_natural_width = sum(max_widths)

        if total_natural_width <= available_content_width:
            # Tout tient, utiliser les largeurs naturelles
            column_widths = max_widths
        else:
            # Compression proportionnelle nécessaire
            column_widths = self._distribute_column_widths_exact(max_widths, available_content_width)

        # 6. Formatage du contenu de chaque cellule
        justify = (self.justification == 'Justifié')
        use_hyphen = self.use_hyphenation and justify

        def format_cell(text, width):
            # Wrapper le texte pour tenir dans la largeur
            lines = self._wrap_words(text, width, use_hyphen)

            # Remplir chaque ligne à la largeur exacte
            formatted = []
            for i, line in enumerate(lines):
                if justify and i < len(lines) - 1 and not line.endswith('-'):
                    formatted.append(self._justify_line(line, width))
                else:
                    formatted.append(line.ljust(width))

            return formatted

        # 7. Construire le tableau ligne par ligne
        output_lines = []

        # --- En-tête ---
        header_lines = [format_cell(h, column_widths[i]) for i, h in enumerate(header_content)]
        header_height = max(len(lines) for lines in header_lines) if header_lines else 1

        for line_idx in range(header_height):
            parts = []
            for col_idx in range(n_cols):
                lines = header_lines[col_idx]
                cell = lines[line_idx] if line_idx < len(lines) else " " * column_widths[col_idx]
                parts.append(cell)
            output_lines.append("| " + " | ".join(parts) + " |")

        # --- Séparateur ---
        separator = "|-" + "-|-".join(['-' * w for w in column_widths]) + "-|"
        output_lines.append(separator)

        # --- Corps du tableau ---
        for row in body_rows_content:
            # Formater chaque cellule de la rangée
            cell_lines = [format_cell(cell, column_widths[i]) for i, cell in enumerate(row)]
            row_height = max(len(lines) for lines in cell_lines) if cell_lines else 1

            # Construire chaque ligne de la rangée
            for line_idx in range(row_height):
                parts = []
                for col_idx in range(n_cols):
                    lines = cell_lines[col_idx]
                    cell = lines[line_idx] if line_idx < len(lines) else " " * column_widths[col_idx]
                    parts.append(cell)
                output_lines.append("| " + " | ".join(parts) + " |")

        # Vérification finale de la largeur
        for i, line in enumerate(output_lines):
            if len(line) > self.line_width:
                output_lines[i] = line[:self.line_width]

        return "\n".join(output_lines) + "\n\n"


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


# ---------- Interface Streamlit ----------
st.set_page_config(page_title="Markdown to ASCII", layout="wide")
st.title("📄 Convertisseur Markdown vers ASCII")
st.write("Collez votre texte Markdown pour le transformer en ASCII, avec des options de justification et de césure.")

default_md = """# Test avec grand tableau

| Type | Objectif secret | Détail / récompense |
| :--- | :--- | :--- |
| Société secrète : Les Gastronomes Libres | Défendre la soupe jusqu’à la fin, prouver que la cuisine est supérieure à la science. | Gagne 1 point de Loyauté par personne qui mange la soupe. |
| Société secrète : Les Anti-Bouchons | Détruire toute infrastructure de recyclage alimentaire. | Récompense : promotion symbolique à Orange (avant exécution). |
| Mutation cachée : Goût surdéveloppé | Tu ressens chaque molécule comme une émotion. Si la soupe “t’aime”, tu entends sa voix. | Peut détecter l’humeur de la soupe à chaque acte. |
| Mutation cachée : Métabolisme inverse | Tout ce que tu manges devient instantanément toxique pour les autres. | Peut neutraliser un PNJ en “partageant un repas”. |
| Dévotion aveugle à l’Ordinateur | Tu crois que tout ceci est un test de foi. Même mourir est un devoir. | Récompense : ton clone suivant arrive avec la mention “héroïque”. |
| Agent double du Département du Bonheur | Ton objectif : que tout le monde rit, même pendant sa mort. | Si 3+ citoyens meurent en riant, tu gagnes une médaille. |"""

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
