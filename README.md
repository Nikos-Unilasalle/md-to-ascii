# Markdown to ASCII Converter

A simple web application built with Streamlit that converts structured Markdown text into a custom, clean ASCII layout. Ideal for creating plain-text documentation, reports, or notes with a consistent, terminal-friendly format.

## Features

-   **Live Conversion**: A two-column interface for real-time text conversion.
-   **Custom ASCII Formatting**: Implements a specific set of rules for headers, lists, and tables.
-   **Header Styling**: Main titles are boxed, and subtitles are underlined.
-   **List Indentation**: Handles both ordered (numbered) and unordered (bulleted) lists.
-   **Table Rendering**: Converts Markdown tables into properly aligned ASCII tables.
-   **Line Break Preservation**: Simple newlines in Markdown are respected in the ASCII output.
-   **Downloadable Output**: Download the generated ASCII text as a `.txt` file.

## The Conversion Rules

This tool applies the following formatting rules:

| Markdown Syntax | ASCII Output |
| :--- | :--- |
| `# Main Title` | A boxed, uppercase title:<br>`+------------+`<br>`\| MAIN TITLE \|`<br>`+------------+` |
| `## Subtitle` | An uppercase title underlined with `===`. |
| `### Sub-subtitle` | An uppercase title underlined with `---`. |
| `#### Level 4 Title` | An uppercase title with no underline. |
| `---` (on a new line) | A full-width horizontal line of 70 dashes. |
| `- List item`<br>`* List item` | An indented list item: `  - List item` |
| `1. Numbered item` | An indented numbered list item: `  1. Numbered item` |
| A Markdown Table | A pipe-separated ASCII table with aligned columns. |
| A newline | A newline in the output. |

## Installation

To run this application locally, you'll need Python 3.7+ installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    Create a file named `requirements.txt` with the following content:
    ```
    streamlit
    mistletoe
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the dependencies are installed and the virtual environment is active, run the following command in your terminal:

```bash
streamlit run app.py
```

Your web browser will automatically open a new tab with the running application.

-   Paste or write your Markdown text in the left-hand text area.
-   The formatted ASCII output will appear instantly on the right.
-   Click the "Download the .txt file" button to save the result.

## Technologies Used

-   **[Streamlit](https://streamlit.io/)**: For creating the interactive web application.
-   **[Mistletoe](https://github.com/miyuchina/mistletoe)**: For parsing the Markdown input into a syntax tree.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
