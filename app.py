import streamlit as st
import mistletoe
import re

# --- Simplified Pre-processor ---
# It does only one thing: replace '---' on its own line with a keyword.
HR_PLACEHOLDER = "-HR-"


def preprocess_markdown(text):
    lines = text.split('\n')
    # Replace any line that is '---' (and nothing else) with our keyword
    processed_lines = [HR_PLACEHOLDER if re.fullmatch(r'-{3,}', line.strip()) else line for line in lines]
    return "\n".join(processed_lines)


# --- Core Conversion Logic ---

def render_ascii(node):
    if node is None: return ""
    node_class_name = node.__class__.__name__

    if node_class_name == 'Heading':
        if not node.children: return ""
        text = "".join(render_ascii(child) for child in node.children).strip().upper()
        # Setext-style title (Text\n---) is also parsed as a Level 2 Heading
        if node.level == 1:
            width = len(text)
            return f"+-{'-' * width}-+\n| {text} |\n+-{'-' * width}-+\n\n"
        elif node.level == 2:
            return f"{text}\n{'=' * len(text)}\n\n"
        elif node.level == 3:
            return f"{text}\n{'-' * len(text)}\n\n"
        else:
            return f"{text}\n\n"

    elif node_class_name == 'Paragraph':
        return "".join(render_ascii(child) for child in node.children) + "\n"

    # HANDLE LINE BREAKS
    elif node_class_name == 'LineBreak':
        return "\n"

    elif node_class_name == 'List':
        output = []
        is_ordered = hasattr(node, 'start') and node.start is not None
        start_num = node.start if is_ordered else 1
        for i, item in enumerate(node.children, start=start_num):
            prefix = f"{i}. " if is_ordered else "- "
            # No extra \n needed, as paragraphs within list items already provide one
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


# --- Main Function ---

def markdown_to_ascii(text):
    if not text or not text.strip(): return ""

    # Manually convert Setext-style headers (e.g., "Title\n---") to standard "## Title"
    # This must be done before the pre-processor handles horizontal rules.
    text = re.sub(r'^(?!#)(.+)\n(-{3,})$', r'## \1', text, flags=re.MULTILINE)

    preprocessed_text = preprocess_markdown(text)
    doc = mistletoe.Document(preprocessed_text)
    raw_output = "".join(render_ascii(child) for child in doc.children)
    
    # Final replacement of the placeholder with the actual horizontal rule
    final_output = raw_output.replace(HR_PLACEHOLDER, '-' * 70)
    
    # Clean up excessive newlines for a tighter output
    final_output = re.sub(r'\n{3,}', '\n\n', final_output)

    return final_output.strip()


# --- Streamlit User Interface ---

st.set_page_config(page_title="Markdown to ASCII Converter", layout="wide")
st.title("📄 Markdown to ASCII Converter")
st.write("Paste your Markdown text below to convert it to the ASCII layout.")

default_md = """# Full Feature Test

This is a paragraph.
It continues on a second line,
and even a third one.

---

Above, a horizontal line should have appeared.

This is a title
---

Above, an underlined level 2 title should have appeared.

## Lists
1. Item one.
2. Item two.
- Bullet A.
- Bullet B.

## Table

| Column 1      | Col 2       | A longer column header |
|---------------|-------------|------------------------|
| Sophie        | Marceau     | Best actress           |
| Nikos         | Priniotakis | Colonel                |

#### END
"""

col1, col2 = st.columns(2)
with col1:
    st.header("Your ugly markdown")
    markdown_input = st.text_area("Enter your text here", height=500, value=default_md)
with col2:
    st.header("ASCII Result")
    if markdown_input:
        ascii_output = markdown_to_ascii(markdown_input)
        st.code(ascii_output, language="")
        st.download_button(
            label="📥 Download .txt file",
            data=ascii_output.encode('utf-8'),
            file_name="output.txt",
            mime="text/plain"
        )
    else:
        st.info("The result will be displayed here once you enter some text.")
