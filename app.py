import streamlit as st
import mistletoe
import re

# --- Pré-processeur et Marqueurs ---
HR_PLACEHOLDER = "-HR-"
# Marqueur pour gérer l'espacement des titres
TITLE_MARKER = "@@TITLE@@"


def preprocess_markdown(text):
    """
    Gère les lignes horizontales et les titres de style Setext avant l'analyse.
    """
    # Gère les titres de style Setext (Titre\n---) en premier
    text = re.sub(r'^(?!#)(.+)\n(-{3,})$', r'## \1', text, flags=re.MULTILINE)

    # Remplace les lignes horizontales par un placeholder
    lines = text.split('\n')
    processed_lines = [HR_PLACEHOLDER if re.fullmatch(r'-{3,}', line.strip()) else line for line in lines]
    return "\n".join(processed_lines)


# --- Cœur de la logique de conversion ---

def render_ascii(node):
    if node is None: return ""
    node_class_name = node.__class__.__name__

    if node_class_name == 'Heading':
        if not node.children: return ""
        text = "".join(render_ascii(child) for child in node.children).strip().upper()

        if node.level == 1:
            width = len(text)
            title_block = f"+-{'-' * width}-+\n| {text} |\n+-{'-' * width}-+"
        elif node.level == 2:
            title_block = f"{text}\n{'=' * len(text)}"
        elif node.level == 3:
            title_block = f"{text}\n{'-' * len(text)}"
        else:
            title_block = f"{text}"

        # On retourne le marqueur suivi du bloc de titre pour post-traitement
        return f"{TITLE_MARKER}{title_block}\n\n"

    elif node_class_name == 'Paragraph':
        return "".join(render_ascii(child) for child in node.children) + "\n"

    elif node_class_name == 'LineBreak':
        return "\n"

    elif node_class_name == 'List':
        output = []
        is_ordered = hasattr(node, 'start') and node.start is not None
        start_num = node.start if is_ordered else 1
        for i, item in enumerate(node.children, start=start_num):
            prefix = f"{i}. " if is_ordered else "- "
            content = "".join(render_ascii(child) for child in item.children).strip()
            output.append(f"  {prefix}{content}")
        return "\n".join(output) + "\n\n"

    elif node_class_name == 'Table':
        def get_cell_content(cell):
            return "".join(render_ascii(child) for child in cell.children).strip()

        if not hasattr(node, 'header') or not node.header.children: return ""
        header_content = [get_cell_content(cell) for cell in node.header.children]
        body_rows_content = [[get_cell_content(cell) for cell in row.children] for row in node.children]
        column_widths = [len(h) for h in header_content]
        for row in body_rows_content:
            for i, cell in enumerate(row):
                if i < len(column_widths): column_widths[i] = max(column_widths[i], len(cell))

        def render_table_row(row_data):
            cells = [content.ljust(column_widths[i]) for i, content in enumerate(row_data)]
            return "| " + " | ".join(cells) + " |"

        separator = "|-" + "-|-".join(['-' * w for w in column_widths]) + "-|"
        output_lines = [render_table_row(header_content), separator]
        for row in body_rows_content: output_lines.append(render_table_row(row))
        return "\n".join(output_lines) + "\n\n"

    elif node_class_name == 'RawText':
        return node.content if hasattr(node, 'content') else ""

    elif hasattr(node, 'children') and node.children is not None:
        return "".join(render_ascii(child) for child in node.children)

    return ""


# --- Fonction principale ---

def markdown_to_ascii(text):
    if not text or not text.strip(): return ""

    preprocessed_text = preprocess_markdown(text)
    doc = mistletoe.Document(preprocessed_text)
    raw_output = "".join(render_ascii(child) for child in doc.children)

    output_with_markers = raw_output.replace(HR_PLACEHOLDER, '-' * 70)

    # CORRECTION FINALE : GESTION DE LA LIGNE VIDE AVANT LES TITRES
    # 1. On remplace n'importe quel espacement (ou absence d'espacement) avant un titre par EXACTEMENT DEUX sauts de ligne.
    spaced_output = re.sub(r'\s*(' + re.escape(TITLE_MARKER) + r')', r'\n\n\1', output_with_markers)
    # 2. On supprime les marqueurs.
    final_output = spaced_output.replace(TITLE_MARKER, '')

    # Nettoyage des sauts de ligne excessifs possibles ailleurs
    final_output = re.sub(r'\n{3,}', '\n\n', final_output)

    return final_output.strip()


# --- Interface utilisateur avec Streamlit ---

st.set_page_config(page_title="Markdown to ASCII", layout="wide")
st.title("📄 Convertisseur Markdown vers ASCII")
st.write("Collez votre texte Markdown pour le transformer en ASCII.")

default_md = """# Test complet des fonctionnalités

Ceci est un paragraphe.
Il continue sur une deuxième ligne.
---
Un titre de niveau 2
---
## Listes
Un paragraphe collé au titre ci-dessus.
1. Élément un.
2. Élément deux.
#### Un Titre sans espace avant
Et un texte juste après.

## Table

| Colonne 1      | Col 2       | Une colonne plus longue |
|---------------|-------------|------------------------|
| Sophie        | Marceau     | Meilleure actrice      |
| Nikos         | Priniotakis | Colonel                |

#### FIN
"""

col1, col2 = st.columns(2)
with col1:
    st.header("Your ugly markdown")
    markdown_input = st.text_area("Entrez votre texte ici", height=500, value=default_md)
with col2:
    st.header("ASCII Result")
    if markdown_input:
        ascii_output = markdown_to_ascii(markdown_input)
        st.code(ascii_output, language="")
        st.download_button(
            label="📥 Télécharger le fichier .txt",
            data=ascii_output.encode('utf-8'),
            file_name="ASCII.txt",
            mime="text/plain"
        )
    else:
        st.info("The result will be displayed here once you enter some text.")
